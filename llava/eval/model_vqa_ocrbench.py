import argparse
import torch
import os
import json
from tqdm import tqdm
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class for OCRBench
class OCRBenchDataset(Dataset):
    def __init__(self, data, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.data = data
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.data[index]
        image_file = line['image_path']
        qs = line['question']
        qs = qs + "\nAnswer the question using a single word or phrase."

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(data, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = OCRBenchDataset(data, image_folder, tokenizer, image_processor, model_config, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,
                                                                           args.model_base,
                                                                           model_name,
                                                                           restore=args.restore)
    # Encoder
    from tore import load_reduction_base
    model = load_reduction_base(model,
                                args.base,
                                args.n_vis,
                                args.restore)

    # Decoder
    model_class_name = type(model).__name__
    if model_class_name == "LlavaLlamaForCausalLM_restore":
        model.model.n_vis = args.n_vis
        model.model.restore = args.restore

    # Load OCRBench data (JSON array)
    with open(os.path.expanduser(args.question_file), "r") as f:
        data = json.load(f)
    data = get_chunk(data, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(data, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, data), total=len(data)):
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Write original data fields + predict as JSONL
        result = {
            "dataset_name": line["dataset_name"],
            "id": line["id"],
            "image_path": line["image_path"],
            "question": line["question"],
            "answers": line["answers"],
            "type": line["type"],
            "predict": outputs,
        }
        ans_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    ans_file.close()


def eval_score(args):
    """Evaluate OCRBench score from merged JSONL result file."""
    results_file = os.path.expanduser(args.results_file)
    data = []
    with open(results_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # Scoring
    for i in range(len(data)):
        answers = data[i]["answers"]
        predict = data[i]['predict']
        dataset_name = data[i]["dataset_name"]
        data[i]['result'] = 0
        if dataset_name == "HME100k":
            if type(answers) == list:
                for j in range(len(answers)):
                    answer = answers[j].strip().replace("\n", " ").replace(" ", "")
                    pred = predict.strip().replace("\n", " ").replace(" ", "")
                    if answer in pred:
                        data[i]['result'] = 1
            else:
                answers_clean = answers.strip().replace("\n", " ").replace(" ", "")
                pred = predict.strip().replace("\n", " ").replace(" ", "")
                if answers_clean in pred:
                    data[i]['result'] = 1
        else:
            if type(answers) == list:
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace("\n", " ")
                    pred = predict.lower().strip().replace("\n", " ")
                    if answer in pred:
                        data[i]['result'] = 1
            else:
                answers_clean = answers.lower().strip().replace("\n", " ")
                pred = predict.lower().strip().replace("\n", " ")
                if answers_clean in pred:
                    data[i]['result'] = 1

    # Print scores
    if len(data) == 1000:
        OCRBench_score = {
            "Regular Text Recognition": 0, "Irregular Text Recognition": 0,
            "Artistic Text Recognition": 0, "Handwriting Recognition": 0,
            "Digit String Recognition": 0, "Non-Semantic Text Recognition": 0,
            "Scene Text-centric VQA": 0, "Doc-oriented VQA": 0,
            "Key Information Extraction": 0, "Handwritten Mathematical Expression Recognition": 0,
        }
        for i in range(len(data)):
            OCRBench_score[data[i]['type']] += data[i]['result']

        recognition_score = (OCRBench_score['Regular Text Recognition']
                             + OCRBench_score['Irregular Text Recognition']
                             + OCRBench_score['Artistic Text Recognition']
                             + OCRBench_score['Handwriting Recognition']
                             + OCRBench_score['Digit String Recognition']
                             + OCRBench_score['Non-Semantic Text Recognition'])
        Final_score = (recognition_score
                       + OCRBench_score['Scene Text-centric VQA']
                       + OCRBench_score['Doc-oriented VQA']
                       + OCRBench_score['Key Information Extraction']
                       + OCRBench_score['Handwritten Mathematical Expression Recognition'])

        print("###########################OCRBench##############################")
        print(f"Text Recognition(Total 300):{recognition_score}")
        print("------------------Details of Recognition Score-------------------")
        print(f"Regular Text Recognition(Total 50): {OCRBench_score['Regular Text Recognition']}")
        print(f"Irregular Text Recognition(Total 50): {OCRBench_score['Irregular Text Recognition']}")
        print(f"Artistic Text Recognition(Total 50): {OCRBench_score['Artistic Text Recognition']}")
        print(f"Handwriting Recognition(Total 50): {OCRBench_score['Handwriting Recognition']}")
        print(f"Digit String Recognition(Total 50): {OCRBench_score['Digit String Recognition']}")
        print(f"Non-Semantic Text Recognition(Total 50): {OCRBench_score['Non-Semantic Text Recognition']}")
        print("----------------------------------------------------------------")
        print(f"Scene Text-centric VQA(Total 200): {OCRBench_score['Scene Text-centric VQA']}")
        print("----------------------------------------------------------------")
        print(f"Doc-oriented VQA(Total 200): {OCRBench_score['Doc-oriented VQA']}")
        print("----------------------------------------------------------------")
        print(f"Key Information Extraction(Total 200): {OCRBench_score['Key Information Extraction']}")
        print("----------------------------------------------------------------")
        print(f"Handwritten Mathematical Expression Recognition(Total 100): {OCRBench_score['Handwritten Mathematical Expression Recognition']}")
        print("----------------------Final Score-------------------------------")
        print(f"Final Score(Total 1000): {Final_score}")
    else:
        AllDataset_score = {"IIIT5K": 0, "svt": 0, "IC13_857": 0, "IC15_1811": 0, "svtp": 0, "ct80": 0,
                            "cocotext": 0, "ctw": 0, "totaltext": 0, "HOST": 0, "WOST": 0, "WordArt": 0,
                            "IAM": 0, "ReCTS": 0, "ORAND": 0, "NonSemanticText": 0, "SemanticText": 0,
                            "STVQA": 0, "textVQA": 0, "ocrVQA": 0, "ESTVQA": 0, "ESTVQA_cn": 0,
                            "docVQA": 0, "infographicVQA": 0, "ChartQA": 0, "ChartQA_Human": 0,
                            "FUNSD": 0, "SROIE": 0, "POIE": 0, "HME100k": 0}
        num_all = {k: 0 for k in AllDataset_score}
        for i in range(len(data)):
            num_all[data[i]['dataset_name']] += 1
            AllDataset_score[data[i]['dataset_name']] += data[i]['result']
        for key in AllDataset_score.keys():
            if num_all[key] > 0:
                print(f"{key}: {AllDataset_score[key] / float(num_all[key])}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    # Inference mode
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    infer_parser.add_argument("--model-base", type=str, default=None)
    infer_parser.add_argument("--image-folder", type=str, default="")
    infer_parser.add_argument("--question-file", type=str, default="./OCRBench.json")
    infer_parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    infer_parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    infer_parser.add_argument("--num-chunks", type=int, default=1)
    infer_parser.add_argument("--chunk-idx", type=int, default=0)
    infer_parser.add_argument("--temperature", type=float, default=0.0)
    infer_parser.add_argument("--top_p", type=float, default=None)
    infer_parser.add_argument("--num_beams", type=int, default=1)
  
    infer_parser.add_argument("--base", type=str, default='HoloV', choices=['DivPrune', 'VisPruner', 'HoloV'])
    infer_parser.add_argument("--n_vis", type=int, default=64, choices=[64, 128, 192])
    infer_parser.add_argument("--restore", action='store_true')

    # Eval mode
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--results-file", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "infer":
        eval_model(args)
    elif args.mode == "eval":
        eval_score(args)
    else:
        parser.print_help()
