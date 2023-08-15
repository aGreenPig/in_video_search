import re
import os
import numpy as np
from numpy.linalg import norm
import gradio as gr
from PIL import Image
import cv2

import torch
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel, AutoModel


i2t_model_id = "nlpconnect/vit-gpt2-image-captioning"
i2t_model = VisionEncoderDecoderModel.from_pretrained(i2t_model_id)
i2t_tokenizer = AutoTokenizer.from_pretrained(i2t_model_id)
i2t_feature_extractor = ViTFeatureExtractor.from_pretrained(i2t_model_id)

ss_model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ss_tokenizer = AutoTokenizer.from_pretrained(ss_model_id)
ss_model = AutoModel.from_pretrained(ss_model_id)

VIDEO_SAMPLE_RATE_SECONDS = 1


def run_in_video_search(video, prompt):
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    descs = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            if frame_count % (fps * VIDEO_SAMPLE_RATE_SECONDS) == 0:
                image = Image.fromarray(frame.astype('uint8'), 'RGB')
                desc = run_image2text(image)
                print(frame_count/fps, "seconds: ", desc)
                descs.append([int(frame_count/fps), desc])
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    sim_scores = run_sentence_similarity(prompt, [x[1] for x in descs])
    ranks = np.argsort([-x for x in sim_scores])
    descs = [descs[i] for i in ranks]
    result = ""
    if len(descs) > 0:
        result = result + "The most matching scene is at second " + \
            str(descs[0][0]) + ": " + descs[0][1]
    if len(descs) > 1:
        result = result + "\nThe second most matching scene is at second " + \
            str(descs[1][0]) + ": " + descs[1][1]
    result = result + \
        "\n\n\nFull log with [second, image/screenshot description] ordered by confidence:\n" + str(
            descs)
    return result


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# returns sentence similarity scores between each element in @candidates and @propmt


def run_sentence_similarity(prompt, candidates):
    encoded_input = ss_tokenizer(
        prompt, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = ss_model(**encoded_input)
    prompt_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask'])

    sim_scores = []
    if isinstance(candidates, str):
        candidates = [candidates]
    for t in candidates:
        encoded_input = ss_tokenizer(
            t, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = ss_model(**encoded_input)
        t_embeddings = mean_pooling(
            model_output, encoded_input['attention_mask'])
        t_embeddings = torch.reshape(t_embeddings, (-1,))
        prompt_embeddings = torch.reshape(prompt_embeddings, (-1,))
        sim_score = np.dot(t_embeddings, prompt_embeddings) / \
            (norm(t_embeddings)*norm(prompt_embeddings))
        sim_scores.append(sim_score)
    print("similarity scores: ", sim_scores)
    return sim_scores

# returns text descriptions given an image


def run_image2text(image):
    img = image.convert('RGB')
    i2t_model.eval()
    pixel_values = i2t_feature_extractor(
        images=[img], return_tensors="pt").pixel_values
    with torch.no_grad():
        output_ids = i2t_model.generate(
            pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

    preds = i2t_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


examples_folder = os.path.join(os.path.dirname(__file__), "examples")
images_folder = os.path.join(examples_folder, "images")
videos_folder = os.path.join(examples_folder, "videos")
image_examples = [os.path.join(images_folder, file)
                  for file in os.listdir(images_folder)]
video_examples = [os.path.join(videos_folder, file)
                  for file in os.listdir(videos_folder)]

# Gradio demo UI configs and actions
with gr.Blocks() as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h2 style="font-weight: 900; font-size: 3rem; margin: 0rem">
        In-Video Search
        </h2>   
        <h2 style="text-align: left; font-weight: 450; font-size: 1rem; margin-top: 2rem; margin-bottom: 1.5rem">
        Locate visual content (with timestamp) within a video by given text prompt.
        <br>
        <br>
        Left side is Image Captioning (Image to Text) demo.
        <br>
        Right side is In-Video Search demo. Try with the video in the /examples folder with search prompt e.g. "pink flowers".
        <br>
        <br>
        Original Github repo <u><a href="https://github.com/aGreenPig/in_video_search/" target="_blank">here</a></u>.
        </h2>
        </div>
        """)

    with gr.Row():
        with gr.Column(scale=1):
            img1 = gr.Image(label="Image to Text/Caption", type='pil')
            button1 = gr.Button(value="Go")
            out1 = gr.Textbox(label="Image Caption")
        with gr.Column(scale=1):
            video3 = gr.Video(label="In-Video search")
            text3 = gr.Textbox(label="Prompt")
            button3 = gr.Button(value="Go")
            out3 = gr.Textbox(label="Output")

    button1.click(run_image2text, inputs=[img1], outputs=[out1])
    button3.click(run_in_video_search, inputs=[video3, text3], outputs=[out3])

    gr.Examples(
        examples=image_examples,
        inputs=img1,
        outputs=out1,
        fn=run_image2text,
        cache_examples=True,
    )
demo.launch(debug=True)
