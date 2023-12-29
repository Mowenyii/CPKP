import os
import sys
import argparse
import torch
from torch.nn import functional as F
from clip.simple_tokenizer import SimpleTokenizer
from clip import clip
import random
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import os
import pdb
import pandas as pd
import numpy as np


def load_clip_to_cpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class Aggregator(torch.nn.Module):
    '''
    Aggregator class
    Mode in ['sum', 'concat', 'neighbor']
    '''

    def __init__(self, batch_size, dim, aggregator):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.eps = 0.01
        self.dim = dim
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=False)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=False)
        self.aggregator = aggregator

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, act, is_agg=False):
        batch_size = user_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        if self.aggregator == 'sum':
            output = (self_vectors + neighbors_agg).view((-1, self.dim))

        elif self.aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.view((-1, 2 * self.dim))

        else:
            output = neighbors_agg.view((-1, self.dim))

        output = self.weights(output)
        return act(output.view((self.batch_size, -1, self.dim)))

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        '''
        This aims to aggregate neighbor vectors
        '''
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.view((self.batch_size, 1, 1, self.dim))

        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim=-1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)

        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim=-1)

        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors).sum(dim=2)

        return neighbors_aggregated


class DataLoader:
    '''
    Data Loader class which makes dataset for training / knowledge graph dictionary
    '''

    def __init__(self, data):
        self.cfg = {
            'music': {
                'item2id_path': './trainers/data/music/item_index2entity_id.txt',
                'kg_path': './trainers/data/music/kg.txt',
                'rating_path': './trainers/data/music/user_artists.dat',
                'rating_sep': '\t',
                'threshold': 0.0
            }
        }
        self.data = data

        df_item2id = pd.read_csv(self.cfg[data]['item2id_path'], sep='\t', header=None, names=['item', 'id'])
        df_kg = pd.read_csv(self.cfg[data]['kg_path'], sep='\t', header=None, names=['head', 'relation', 'tail'])
        df_rating = pd.read_csv(self.cfg[data]['rating_path'], sep=self.cfg[data]['rating_sep'],
                                names=['userID', 'itemID', 'rating'], skiprows=1)

        # df_rating['itemID'] and df_item2id['item'] both represents old entity ID
        df_rating = df_rating[df_rating['itemID'].isin(df_item2id['item'])]
        df_rating.reset_index(inplace=True, drop=True)

        self.df_item2id = df_item2id
        self.df_kg = df_kg
        self.df_rating = df_rating

        self.user_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()

        self._encoding()

    def _encoding(self):
        '''
        Fit each label encoder and encode knowledge graph
        '''
        self.user_encoder.fit(self.df_rating['userID'])
        # df_item2id['id'] and df_kg[['head', 'tail']] represents new entity ID
        self.entity_encoder.fit(pd.concat([self.df_item2id['id'], self.df_kg['head'], self.df_kg['tail']]))
        self.relation_encoder.fit(self.df_kg['relation'])

        # encode df_kg
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'])
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation']) + 1

    def _build_dataset(self):
        '''
        Build dataset for training (rating data)
        It contains negative sampling process
        '''
        print('Build dataset dataframe ...', end=' ')
        # df_rating update
        df_dataset = pd.DataFrame()
        df_dataset['userID'] = self.user_encoder.transform(self.df_rating['userID'])

        # update to new id
        item2id_dict = dict(zip(self.df_item2id['item'], self.df_item2id['id']))
        self.df_rating['itemID'] = self.df_rating['itemID'].apply(lambda x: item2id_dict[x])
        df_dataset['itemID'] = self.entity_encoder.transform(self.df_rating['itemID'])
        df_dataset['label'] = self.df_rating['rating'].apply(lambda x: 0 if x < self.cfg[self.data]['threshold'] else 1)

        # negative sampling
        df_dataset = df_dataset[df_dataset['label'] == 1]
        # df_dataset requires columns to have new entity ID
        full_item_set = set(range(len(self.entity_encoder.classes_)))
        user_list = []
        item_list = []
        label_list = []
        for user, group in df_dataset.groupby(['userID']):
            item_set = set(group['itemID'])
            negative_set = full_item_set - item_set
            negative_sampled = random.sample(negative_set, len(item_set))
            user_list.extend([user] * len(negative_sampled))
            item_list.extend(negative_sampled)
            label_list.extend([0] * len(negative_sampled))
        negative = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'label': label_list})
        df_dataset = pd.concat([df_dataset, negative])

        df_dataset = df_dataset.sample(frac=1, replace=False, random_state=999)
        df_dataset.reset_index(inplace=True, drop=True)
        print('Done')
        return df_dataset

    def _construct_kg(self):
        '''
        Construct knowledge graph
        Knowledge graph is dictionary form
        'head': [(relation, tail), ...]
        '''
        print('Construct knowledge graph ...', end=' ')
        kg = dict()
        for i in range(len(self.df_kg)):
            head = self.df_kg.iloc[i]['head']
            relation = self.df_kg.iloc[i]['relation']
            tail = self.df_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        print('Done')
        return kg

    def load_dataset(self):
        return self._build_dataset()

    def load_kg(self):
        return self._construct_kg()

    def get_encoders(self):
        return (self.user_encoder, self.entity_encoder, self.relation_encoder)

    def get_num(self):
        return (len(self.user_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_))


class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label



food101_item_idss=torch.tensor([1260,2593,3120,3120,3120,713,2896,3226,1246,407,2593,3120,1260,3120,713,2896,3226,1246,407,2593,3120,3120,3120,713,2896,3226,1246,1260,2593,3120,3120,3120,713,2896,3226,1246,407,2593,3120,3120,1260,713,2896,3226,1246,407,2593,1260,3120,3120,713,2896,3226,1246,
                        407,1260,3120,3120,3120,713,2896,3226,1246,407,2593,3120,3120,3120,713,2896,3226,1246,407,2593,3120,1260,3120,713,2896,3226,1246,407,2593,3120,3120,3120,713,2896,3226,1246,407,2593,3120,3120,3120,713,1260,3226,1246,3226,1246])


parser = argparse.ArgumentParser()
parser.add_argument("fpath", type=str, help="Path to the learned prompt")
parser.add_argument("topk", type=int, help="Select top-k similar words")
parser.add_argument("dataset", type=str, help="dataset")
args = parser.parse_args()

fpath = args.fpath
topk = args.topk
dataset_name=args.dataset

class KGCN(torch.nn.Module):
    def __init__(self,dataset_name,  num_user, num_ent, num_rel, kg, device):
        super(KGCN, self).__init__()
        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = 1
        N_CTX=16
        CSC=False
        DATASET = dataset_name

        if DATASET == "oxford_flowers":
            self.batch_size = 102
        elif DATASET == "food101":
            self.batch_size = 101
        elif DATASET == "fgvc_aircraft":
            self.batch_size = 100
        elif DATASET == "eurosat":
            self.batch_size = 10
        elif DATASET == "ucf101":
            self.batch_size = 101
        elif DATASET == "oxford_pets":
            self.batch_size = 37
        elif DATASET == "caltech101":
            self.batch_size = 100
        elif (
                DATASET == "imagenet" or DATASET == "imagenetv2" or DATASET == "imagenet_sketch"):
            self.batch_size = 1000
        elif (DATASET == "imagenet_a" or DATASET == "imagenet_r"):
            self.batch_size = 200
        elif DATASET == "stanford_cars":
            self.batch_size = 196
        elif DATASET == "dtd":
            self.batch_size = 47
        elif DATASET == "sun397":
            self.batch_size = 397
        self.dim = 512
        self.n_neighbor = 8

        self.kg = kg

        self.device = device
        self.fea_agg = None
        self.fea = None
        self.aggregator = Aggregator(self.batch_size, self.dim, 'sum')

        self._gen_adj()

        self.usr = torch.nn.Embedding(num_user, self.dim)
        self.ent = torch.nn.Embedding(num_ent, self.dim)
        self.rel = torch.nn.Embedding(num_rel + 1, self.dim)

        self.isLinear = False
        if (N_CTX == 16 or N_CTX == 4) and CSC == False:
            self.isLinear = True
            self.Linear = torch.nn.Linear(self.batch_size, N_CTX, bias=False)

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)

        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)

            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])

    def forward(self, u, v, r=None, input_entities=None, input_relations=None):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        self.fea_agg = None
        batch_size = u.size(0)

        if batch_size != self.batch_size:
            self.batch_size = batch_size


        u = u.view((-1, 1))
        v = v.view((-1, 1))

        # [batch_size, dim]
        user_embeddings = self.usr(u).squeeze(dim=1)
        if input_entities == None and input_relations == None:
            entities, relations = self._get_neighbors(v)  # [class_num,1]
        else:
            entities, relations = input_entities, input_relations


        if r != None:
            for i in range(self.batch_size):
                for j in range(self.n_neighbor):
                    if relations[0][i][j] == r:
                        entities[1][i][j] = entities[0][i]
                        relations[0][i][j] = 0


        item_embeddings = self._aggregate(user_embeddings, entities, relations)  # [cls_num,dim]
        agg_flag = True
        if r == None and input_entities == None and input_relations == None and agg_flag:
            item_embeddings_agg = self._aggregate(user_embeddings, entities, relations, True)

        if self.isLinear == False:
            fea = self.Linear(item_embeddings.t()).t()
            if r == None and input_entities == None and input_relations == None and agg_flag:
                self.fea_agg = self.Linear(item_embeddings_agg.t()).t()
        else:
            fea = torch.unsqueeze(item_embeddings, 2).permute(0, 2, 1)
            if r == None and input_entities == None and input_relations == None and agg_flag:
                self.fea_agg = torch.unsqueeze(item_embeddings_agg, 2).permute(0, 2, 1)

        self.fea = fea
        return fea

    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).view((self.batch_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    def _aggregate(self, user_embeddings, entities, relations, is_agg=False):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act,
                    is_agg=is_agg
                )

                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].view((self.batch_size, self.dim))


class KGCN_all(torch.nn.Module):
    def __init__(self,dataset_name,num_user, num_entity, num_relation, kg):
        super(KGCN_all, self).__init__()
        self.KGCN=KGCN(dataset_name,num_user, num_entity, num_relation, kg, torch.device('cuda'))

    def forward(self,item_idss,entities_list_0,entities_list_1,relations_list):
        return self.KGCN(item_idss, item_idss,None,[entities_list_0,entities_list_1],[relations_list])#self.KGCN(user_ids, item_ids,None,entities_list,relations_list)



if dataset_name == "food101":
    item_idss=food101_item_idss
    weight=1e-1




assert os.path.exists(fpath)

print(f"Return the top-{topk} matched words")

tokenizer = SimpleTokenizer()
clip_model = load_clip_to_cpu()
token_embedding = clip_model.token_embedding.weight
print(f"Size of token embedding: {token_embedding.shape}")



prompt_learner = torch.load(fpath, map_location="cpu")["state_dict"]

ctx = prompt_learner["ctx"]

ctx = ctx.float()

data_loader = DataLoader('music')
print("||||||||||||||||||||||||||||||||wiki-data||||||||||||||||||||||")
kg = data_loader.load_kg()
num_user, num_entity, num_relation = data_loader.get_num()


# pdb.set_trace()
KGCN_fn= KGCN_all(dataset_name,num_user, num_entity, num_relation, kg)
KGCN_fn.load_state_dict(prompt_learner, strict=False)

entities_list_0=prompt_learner['KGCN_entities_0'].unsqueeze(1)
entities_list_1=prompt_learner['KGCN_entities_1']
relations_list=prompt_learner["KGCN_relations_0"]
KGCN_fn.eval()
with torch.no_grad():
    outputs = KGCN_fn(item_idss,entities_list_0,entities_list_1,relations_list)

ctx = (ctx +  weight*outputs)
print(f"Size of context: {ctx.shape}")

if ctx.dim() == 2:
    # Generic context
    distance = torch.cdist(ctx, token_embedding)
    print(f"Size of distance matrix: {distance.shape}")
    sorted_idxs = torch.argsort(distance, dim=1)
    sorted_idxs = sorted_idxs[:, :topk]

    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.decoder[idx.item()] for idx in idxs]
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        print(f"{words[0]} ({dist[0]})")  

elif ctx.dim() == 3:
    # Class-specific context
    # raise NotImplementedError
    ctx =torch.squeeze(ctx)
    # Generic context
    distance = torch.cdist(ctx, token_embedding)
    print(f"Size of distance matrix: {distance.shape}")
    sorted_idxs = torch.argsort(distance, dim=1)
    sorted_idxs = sorted_idxs[:, :topk]

    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.decoder[idx.item()] for idx in idxs]
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        # print(f"{m+1}: {words} {dist}")  
        print(f"{words[0]} ({dist[0]})")  
