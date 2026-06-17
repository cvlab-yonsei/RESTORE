import torch

# divprune
def pairwise_cosine_similarity(matrix):
    norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
    cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
    return cosine_similarity

def DivPrune(
    visual_feature_vectors, 
    image_feature_length, 
    cosine_matrix=None, 
    # threshold_ratio=0.1,
    threshold_n=64,
):
    threshold_terms = threshold_n
    
    if cosine_matrix is None:
        cosine_matrix = 1.0 - (pairwise_cosine_similarity(visual_feature_vectors))

    s = torch.empty(threshold_terms, dtype=torch.long, device=visual_feature_vectors.device)
    for i in range(threshold_terms):
        if i==0:
            m2 = cosine_matrix
        else:
            m2 = torch.index_select(cosine_matrix, 0, torch.index_select(s,0,torch.arange(0,i,device=cosine_matrix.device)))

        if i==0:
            scores = torch.topk(m2, 2,dim=0,largest=False).values[1,:] #for distance
        else:
            scores = torch.min(m2, dim=0).values #for distance: Eqn (3)

        phrase_to_add_idx = torch.argmax(scores)
        s[i] = phrase_to_add_idx
        
    return s
