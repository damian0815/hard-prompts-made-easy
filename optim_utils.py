import random
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from statistics import mean
import copy
import json
from typing import Any, Mapping

import open_clip

import torch

from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)

from tqdm.auto import tqdm

max_len = 77
bos_token_id = 49406
eos_token_id = 49407


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def nn_project(curr_embeds, embedding_layer, print_hits=False):
    with torch.no_grad():
        bsz,seq_len,emb_dim = curr_embeds.shape
        
        # Using the sentence transformers semantic search which is 
        # a dot product exact kNN search between a set of 
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1,emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds) # queries

        embedding_matrix = embedding_layer.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)
        
        hits = semantic_search(curr_embeds, embedding_matrix, 
                                query_chunk_size=curr_embeds.shape[0], 
                                top_k=1,
                                score_function=dot_score)

        if print_hits:
            all_hits = []
            for hit in hits:
                all_hits.append(hit[0]["score"])
            print(f"mean hits:{mean(all_hits)}")
        
        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
        nn_indices = nn_indices.reshape((bsz,seq_len))

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def decode_ids(input_ids, tokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()

    texts = []

    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))

            texts.append('|'.join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))

    return texts


def download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


def get_target_features(model, preprocess, tokenizer_funct, device, target_images=None, target_prompts=None):
    all_target_features = None
    if target_images is not None:
        with torch.no_grad():
            curr_images = [preprocess(i).unsqueeze(0) for i in target_images]
            curr_images = torch.concatenate(curr_images).to(device)
            all_target_features = model.encode_image(curr_images)
            
    if target_prompts is not None:
        texts = tokenizer_funct(target_prompts).to(device)
        text_features = model.encode_text(texts)
        #print(text_features.shape, all_target_features.shape)
        all_target_features = text_features if all_target_features is None else torch.cat([all_target_features, text_features])
        #print("->", all_target_features.shape)

    return all_target_features


def initialize_prompt(tokenizer, token_embedding, args, device, 
                      initial_prompt=None, 
                      prompt_template="{}"):

    bos_text = "<start_of_text>"

    
    template_parts = prompt_template.split("{}")
    if len(template_parts) != 2:
        raise ValueError(f"invalid promt_template '{prompt_template}' - must contain " + "{}")
    prefix = template_parts[0].strip()
    suffix = template_parts[1].strip()

    prefix_ids = tokenizer.encode(prefix)
    suffix_ids = tokenizer.encode(suffix)
    #print(f"prompt template '{prompt_template}' -> prefix ids {prefix_ids}, suffix ids {suffix_ids}")
    
    if initial_prompt is not None:
        initial_ids = tokenizer.encode(initial_prompt)
        prompt_len = len(initial_ids)
        prompt_ids = torch.tensor([[bos_token_id] + prefix_ids + initial_ids + suffix_ids + [eos_token_id]] * args.prompt_bs).to(device)
    else:
        prompt_len = min(args.prompt_len, max_len-2)
        prompt_ids = torch.stack([torch.cat(
            [torch.tensor([bos_token_id] + prefix_ids), 
             torch.randint(len(tokenizer.encoder), (prompt_len,)), 
             torch.tensor(suffix_ids + [eos_token_id])
            ]) for i in range(args.prompt_bs)]).to(device)
        #prompt_ids = torch.randint(len(tokenizer.encoder), (args.prompt_bs, prompt_len)).to(device)
    
    #prompt_ids = prefix_ids + initial_ids + suffix_ids
    prompt_embeds = token_embedding(prompt_ids).detach()
    #print(f"-> prompt ids {prompt_ids} -> embeds {prompt_embeds}")
    prompt_embeds.requires_grad = True

    dummy_ids = [bos_token_id] + prefix_ids + (prompt_len * [-1]) + suffix_ids + [eos_token_id]
    assert len(dummy_ids) <= max_len
    dummy_ids += [0] * (max_len - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * args.prompt_bs).to(device)


    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False
    #print("dummy_ids:", dummy_ids, "\ntmp_dummy_ids:", tmp_dummy_ids, "\ndummy embeds:", dummy_embeds)
    
    return prompt_embeds, dummy_embeds, dummy_ids

    """
    # randomly optimize prompt embeddings
    prompt_ids = torch.randint(len(tokenizer.encoder), (args.prompt_bs, prompt_len)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()
    prompt_embeds.requires_grad = True

    # initialize the template
    if initial_prompt is None:
        template_text = "{}"
        padded_template_text = template_text.format(" ".join([bos_text] * prompt_len))
        print("padded_template_text:", padded_template_text)
        dummy_ids = tokenizer.encode(padded_template_text)
    else:
        
        dummy_ids = tokenizer.encode(initial_prompt)
    print("tokenized initial prompt: ", dummy_ids)

    
    
    # -1 for optimized tokens
    dummy_ids = [i if i != bos else -1 for i in dummy_ids]
    dummy_ids = [bos] + dummy_ids + [eos]
    dummy_ids += [0] * (max_len - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * args.prompt_bs).to(device)
    """



def optimize_prompt_loop(model, tokenizer, token_embedding,
                         all_target_features, args, device,
                         all_target_loss_scales=None, 
                         initial_prompt=None,
                         prompt_template='{}',
                        noise_amount=0):
    if all_target_loss_scales is None:
        all_target_loss_scales = torch.ones(all_target_features.shape[0])
    else:
        assert all_target_loss_scales.shape[0] == all_target_features.shape[0]
    opt_iters = args.iter
    lr = args.lr
    weight_decay = args.weight_decay
    print_step = args.print_step
    batch_size = args.batch_size
    print_new_best = getattr(args, 'print_new_best', False)

    # initialize prompt
    prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(tokenizer, token_embedding, args, device, 
                                                               initial_prompt=initial_prompt,
                                                               prompt_template=prompt_template
                                                              )
    p_bs, p_len, p_dim = prompt_embeds.shape

    # get optimizer
    input_optimizer = torch.optim.AdamW([prompt_embeds], lr=lr, weight_decay=weight_decay)

    best_count = 10
    best = [None] * best_count

    def decode_embeds(embeds):
        _, indices = nn_project(embeds, token_embedding, print_hits=False)
        return indices, decode_ids(indices, tokenizer)


    pbar = tqdm(range(opt_iters))
    for step in pbar:
        try:
            # randomly sample sample images and get features
            if batch_size is None or batch_size > len(all_target_features):
                batch_size = len(all_target_features)
                permutation = torch.arange(len(all_target_features))
            else:
                permutation = torch.randperm(len(all_target_features))
            target_features = all_target_features[permutation][0:batch_size]
            target_loss_scales = all_target_loss_scales[permutation][0:batch_size]
                
            universal_target_features = all_target_features
            universal_loss_scales = all_target_loss_scales
            
            # forward projection
            projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding, print_hits=False)
            # sanity check
            assert nn_indices[:, 0] == bos_token_id
            assert nn_indices[:, -1] == eos_token_id
            
            #print("forward projection got", nn_indices)
            proj_len = projected_embeds.shape[1]
            target_train_indices = dummy_ids == -1
            source_train_indices = target_train_indices[:, :proj_len]
            
            if noise_amount > 0:
                projected_embeds[source_train_indices] += torch.randn_like(projected_embeds[source_train_indices]) * noise_amount

            # get cosine similarity score with all target features
            with torch.no_grad():
                # padded_embeds = copy.deepcopy(dummy_embeds)
                padded_embeds = dummy_embeds.detach().clone()
                #print(dummy_ids.shape)
                #print(projected_embeds.shape, padded_embeds.shape, projected_embeds.reshape(-1, p_dim).shape)
                #print("overwrite", target_train_indices)
                #print(padded_embeds[target_train_indices].shape)
                #print("vindaloo")
                #print(projected_embeds[source_train_indices].shape)
                #print("huh")
                padded_embeds[target_train_indices] = projected_embeds[source_train_indices].reshape(-1, p_dim)
                #print("evaluating score for:", decode_embeds(padded_embeds))    
                logits_per_image, _ = model.forward_text_embedding(padded_embeds, dummy_ids, universal_target_features)
                #print(logits_per_image.device, universal_loss_scales.device)
                logits_per_image_scaled = logits_per_image * universal_loss_scales.unsqueeze(0).t()
                #print("logits_per_image:", logits_per_image, ", scaled:", logits_per_image_scaled)
                scores_per_prompt = logits_per_image.mean(dim=0)
                #print("scores:", scores_per_prompt)       
                
                universal_cosim_score = scores_per_prompt.max().item()
                best_indx = scores_per_prompt.argmax().item()
                #print("scores per prompt:", scores_per_prompt, "-> best:", universal_cosim_score, best_indx)
            
            # tmp_embeds = copy.deepcopy(prompt_embeds)
            tmp_embeds = prompt_embeds.detach().clone()
            tmp_embeds.data = projected_embeds.data
            tmp_embeds.requires_grad = True
            
            # padding
            # padded_embeds = copy.deepcopy(dummy_embeds)
            padded_embeds = dummy_embeds.detach().clone()
            padded_embeds[dummy_ids == -1] = tmp_embeds[source_train_indices].reshape(-1, p_dim)

            #print("doing loss with current padded:", decode_embeds(padded_embeds))    

            logits_per_image, _ = model.forward_text_embedding(padded_embeds, dummy_ids, target_features)
            cosim_scores = logits_per_image
            #print("cosine similarity:", cosim_scores)
            codiff_scores = 1 - cosim_scores.mean(dim=1)
            #print("cosine diff:", codiff_scores)
            codiff_scores_weighted = codiff_scores * target_loss_scales.to(codiff_scores.device)
            #print("weighted cosine diff:", codiff_scores_weighted)

            loss = codiff_scores_weighted.mean() * args.loss_weight
            
            """
            loss = 1 - cosim_scores.mean(dim=1)
            #print("loss in:", loss)
            loss = loss * target_loss_scales.to(loss.device)
            #print("loss after 1:", loss)
            loss = loss.mean() * args.loss_weight
            #print("loss after 2:", loss)
            """

            prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])
            
            input_optimizer.step()
            input_optimizer.zero_grad()
    
            curr_lr = input_optimizer.param_groups[0]["lr"]
            #cosim_scores = cosim_scores.mean().item()

            # decode text, without <bos> and <eos> markers
            decoded_text = decode_ids(nn_indices[:, 1:-1], tokenizer)[best_indx]
            pbar.set_postfix({'loss': loss.item(), 'curr': decoded_text}, )
            if print_step is not None and (step % print_step == 0 or step == opt_iters-1):
                per_step_message = f"step: {step}, lr: {curr_lr}"
                if not print_new_best:
                    per_step_message = f"\n{per_step_message}, cosim: {universal_cosim_score:.3f}, text: {decoded_text}"
                print(per_step_message)

            # update bests list if we see a new prompt
            if not any(b is not None and b[1].strip() == decoded_text.strip() for b in best):
                prev_best = best
                best = sorted(best + [(universal_cosim_score, decoded_text)], 
                              key=lambda x: -1000 if x is None else x[0], 
                              reverse=True
                         )[:best_count]
                if best != prev_best and print_new_best:
                    print("new best cosine sim / prompt:")
                    for sim_and_text in [b for b in best if b is not None]:
                        print(f"{'*' if sim_and_text[1] == decoded_text else ' '} {sim_and_text[0]:.3f}: {sim_and_text[1]}")

        except KeyboardInterrupt:
            break


    best_sim = best[0][0]
    best_text = best[0][1]
    if print_step is not None:
        print()
        print(f"best cosine sim: {best_sim}")
        print(f"best prompt: {best_text}")
        print(f"top {len(best)}:")
        print(best)

    return best_text



def optimize_prompt(model, preprocess, args, device, 
                    target_images=None,
                    target_prompts=None, 
                    target_loss_scales=None,
                    initial_prompt=None,
                    prompt_template='{}',
                    noise_amount=0,
                    ):
    token_embedding = model.token_embedding
    tokenizer = open_clip.tokenizer._tokenizer
    tokenizer_funct = open_clip.get_tokenizer(args.clip_model)


    # get target features
    all_target_features = get_target_features(model, preprocess, tokenizer_funct, device, 
                                             target_images=target_images,
                                             target_prompts=target_prompts
                                             )

    all_target_loss_scales = torch.ones(len(all_target_features)) if target_loss_scales is None else target_loss_scales
    all_target_loss_scales = all_target_loss_scales.to(device)
    
    # optimize prompt
    learned_prompt = optimize_prompt_loop(model, tokenizer, token_embedding, 
                                          all_target_features, 
                                           args, device,
                                         all_target_loss_scales=all_target_loss_scales,
                                         initial_prompt=initial_prompt,
                                          prompt_template=prompt_template,
                                         noise_amount=noise_amount)

    return learned_prompt
    

def measure_similarity(orig_images, images, ref_model, ref_clip_preprocess, device):
    with torch.no_grad():
        ori_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in orig_images]
        ori_batch = torch.concatenate(ori_batch).to(device)

        gen_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in images]
        gen_batch = torch.concatenate(gen_batch).to(device)
        
        ori_feat = ref_model.encode_image(ori_batch)
        gen_feat = ref_model.encode_image(gen_batch)
        
        ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
        
        return (ori_feat @ gen_feat.t()).mean().item()
        