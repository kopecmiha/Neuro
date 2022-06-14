import time
import torch.backends.cudnn as cudnn

from matplotlib import pyplot as plt
cudnn.benchmark = True
from model import *
import warnings
from PIL import Image, ImageEnhance

warnings.filterwarnings("ignore", category=UserWarning)  # get rid of interpolation warning
from util import display_image, load_model, post_process_generator_output, load_source

start_time = time.time()
device = 'cuda'  # @param ['cuda', 'cpu']

generator = Generator(256, 512, 8, channel_multiplier=2).eval().to(device)
generator2 = Generator(256, 512, 8, channel_multiplier=2).eval().to(device)

mean_latent1 = load_model(generator, 'face.pt')
mean_latent2 = load_model(generator2, 'disney.pt')
truncation = .5

face_seed = 70870  # @param {type:"number"}
disney_seed = 50000  # @param {type:"number"}

plt.rcParams['figure.dpi'] = 150

with torch.no_grad():
    torch.manual_seed(face_seed)
    source_code = torch.randn([1, 512]).to(device)
    latent1 = generator.get_latent(source_code, truncation=truncation, mean_latent=mean_latent1)
    #latent1 = load_source(['me'], generator, device)
    source_im, _ = generator(latent1)
    print(source_im)
    torch.manual_seed(disney_seed)
    reference_code = torch.randn([1, 512]).to(device)
    latent2 = generator2.get_latent(reference_code, truncation=truncation, mean_latent=mean_latent2)
    reference_im, _ = generator2(latent2)
    image = display_image(torch.cat([source_im, reference_im], -1), size=None, title='Input/Reference')
    image = post_process_generator_output(image)
    Image.fromarray(np.uint8(image), 'RGB').save("source.jpg")

num_swap = 6
alpha = 0.5

early_alpha = 0

with torch.no_grad():
    noise1 = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
    noise2 = [getattr(generator2.noises, f'noise_{i}') for i in range(generator2.num_layers)]

    out1 = generator.input(latent1[0])
    out2 = generator2.input(latent2[0])
    out = (1 - early_alpha) * out1 + early_alpha * out2

    out1, _ = generator.conv1(out, latent1[0], noise=noise1[0])
    out2, _ = generator2.conv1(out, latent2[0], noise=noise2[0])
    out = (1 - early_alpha) * out1 + early_alpha * out2
    #     out = out2

    skip1 = generator.to_rgb1(out, latent1[1])
    skip2 = generator2.to_rgb1(out, latent2[1])
    skip = (1 - early_alpha) * skip1 + early_alpha * skip2

    i = 2
    for conv1_1, conv1_2, noise1_1, noise1_2, to_rgb1, conv2_1, conv2_2, noise2_1, noise2_2, to_rgb2 in zip(
            generator.convs[::2], generator.convs[1::2], noise1[1::2], noise1[2::2], generator.to_rgbs,
            generator2.convs[::2], generator2.convs[1::2], noise2[1::2], noise2[2::2], generator2.to_rgbs
    ):
        conv_alpha = early_alpha if i < num_swap else alpha
        out1, _ = conv1_1(out, latent1[i], noise=noise1_1)
        out2, _ = conv2_1(out, latent2[i], noise=noise2_1)
        out = (1 - conv_alpha) * out1 + conv_alpha * out2
        #         out = out1
        i += 1

        conv_alpha = early_alpha if i < num_swap else alpha
        out1, _ = conv1_2(out, latent1[i], noise=noise1_2)
        out2, _ = conv2_2(out, latent2[i], noise=noise2_2)
        out = (1 - conv_alpha) * out1 + conv_alpha * out2
        #         out = out1
        i += 1

        conv_alpha = early_alpha if i < num_swap else alpha
        skip1 = to_rgb1(out, latent1[i], skip)
        skip2 = to_rgb2(out, latent2[i], skip)
        skip = (1 - conv_alpha) * skip1 + conv_alpha * skip2

        i += 1
    image = skip.clamp(-1, 1)
    image = display_image(image)
    image = post_process_generator_output(image)
    image = Image.fromarray(np.uint8(image), 'RGB')
    enhancer = ImageEnhance.Brightness(image).enhance(1.6)
    enhancer.save("result.jpg")

print(time.time() - start_time)
