import copy
import json
import os
import warnings

import torch
from absl import app, flags
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import wandb

from diffusion import GaussianDiffusion_distillation_Trainer, GaussianDiffusionTrainer, GaussianDiffusionSampler, distillation_cache_Trainer, GaussianDiffusion_joint_Sampler
from model import UNet, UNetWithFeatures
from score.both import get_inception_and_fid_score

from diffusers import DDIMPipeline, DDPMScheduler
import torch

FLAGS = flags.FLAGS

flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
flags.DEFINE_bool('distill', False, help='perform knowledge distillation')

# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 1e-5, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_bool('distill_features', False, help='perform knowledge distillation using intermediate features')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 32, "sampling size of images")
flags.DEFINE_integer('sample_step', 10000, help='frequency of sampling')
# WandB 관련 FLAGS 추가
flags.DEFINE_string('wandb_project', 'distill_caching_ddpm', help='WandB project name')
flags.DEFINE_string('wandb_run_name', None, help='WandB run name')
flags.DEFINE_string('wandb_notes', '', help='Notes for the WandB run')

# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 100000, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
#Caching
flags.DEFINE_integer('cache_n', 64, help='size of caching data per timestep')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def prepare_dataloader():
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    return dataloader



def visualize_t_cache_distribution(t_cache):
    # CPU로 이동하여 numpy 배열로 변환
    t_cache_cpu = t_cache.cpu().numpy()

    # 히스토그램을 그려 분포 확인 (bin 수를 1000으로 설정)
    plt.figure(figsize=(12, 6))
    plt.hist(t_cache_cpu, range=(0, 1000), bins=1000, alpha=0.7, color='blue')
    plt.title('Distribution of t_cache')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.ylim(0, 1000)
    plt.grid(True)

    # 저장할 디렉토리가 없다면 생성
    os.makedirs('./cache_test', exist_ok=True)

    plt.savefig('./cache_test/temp_frame.png')
    plt.close()

def log_gpu_usage():
    # 현재 사용 중인 메모리와 예약된 메모리를 출력
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


def distill_caching_random():

    # Initialize WandB
    wandb.init(
        project=FLAGS.wandb_project,
        name=FLAGS.wandb_run_name,
        notes=FLAGS.wandb_notes,
        config={
            "learning_rate": FLAGS.lr,
            "architecture": "UNet",
            "dataset": "your_dataset_name",
            "epochs": FLAGS.total_steps,
        }
    )


    # Load pretrained teacher model
    teacher_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    teacher_model.load_state_dict(ckpt['ema_model'])
    teacher_model.eval()

    # Initialize student model
    student_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    
    ddim_pipeline = DDIMPipeline(
    unet=teacher_model,  # 학습된 UNet 모델
    scheduler=DDPMScheduler()  # DDIM 방식으로 샘플링하기 위한 스케줄러
    ).to("cuda")

    optim = torch.optim.Adam(student_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    trainer = distillation_cache_Trainer(
        teacher_model, student_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T,
        FLAGS.mean_type, FLAGS.var_type, FLAGS.distill_features).to(device)
    joint_sampler = GaussianDiffusion_joint_Sampler(
        teacher_model, student_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    student_sampler = GaussianDiffusionSampler(
        student_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    

    ###### prepare cache ######
    
    img_cache = torch.randn(FLAGS.cache_n*1000, 3, FLAGS.img_size, FLAGS.img_size).to(device)
    t_cache = torch.ones(FLAGS.cache_n*1000, dtype=torch.long, device=device)*(FLAGS.T-1)

    with torch.no_grad():
        for i in range(FLAGS.T):
            start_time = time.time()
            
            start_idx = (i * FLAGS.cache_n)
            end_idx = start_idx + FLAGS.cache_n

            x_t = img_cache[start_idx:end_idx]
            t = t_cache[start_idx:end_idx]

            img_cache[start_idx:end_idx] = trainer.teacher_sampling(x_t, i)
            t_cache[start_idx:end_idx] = torch.ones(FLAGS.cache_n, dtype=torch.long, device=device)*(i)
            print(f"start_idx: {start_idx}, end_idx: {end_idx}")
            print(t_cache)

            elapsed_time = time.time() - start_time
            print(f"Iteration {i + 1}/{FLAGS.T} completed in {elapsed_time:.2f} seconds.")

            visualize_t_cache_distribution(t_cache)

    ##################################

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()

            # Step 2: Randomly sample from img_cache and t_cache without shuffling
            indices = torch.randint(0, img_cache.size(0), (FLAGS.batch_size,), device=device)

            # Sample img_cache and t_cache using the random indices
            x_t = img_cache[indices]
            t = t_cache[indices]

            # Calculate distillation loss
            output_loss, x_t_, total_loss = trainer(x_t, t)

            # Backward and optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()

            ### cache update ###
            img_cache[indices] = x_t_
            t_cache[indices] -= 1
            
            num_999 = torch.sum(t_cache == (FLAGS.T - 1)).item()

            if num_999 < FLAGS.cache_n:
                missing_999 = FLAGS.cache_n - num_999
                non_999_indices = (t_cache != (FLAGS.T - 1)).nonzero(as_tuple=True)[0]
                random_indices = torch.randperm(non_999_indices.size(0), device=device)[:missing_999]
                selected_indices = non_999_indices[random_indices]
                t_cache[selected_indices] = FLAGS.T - 1
                img_cache[selected_indices] = torch.randn(missing_999, 3, FLAGS.img_size, FLAGS.img_size, device=device)

            # t_cache에서 값이 0인 인덱스를 찾아 초기화
            zero_indices = (t_cache < 0).nonzero(as_tuple=True)[0]
            num_zero_indices = zero_indices.size(0)

            # 0인 인덱스가 있는 경우에만 초기화 수행
            if num_zero_indices > 0:
                # 0인 인덱스를 1에서 FLAGS.T-1 사이의 랜덤한 정수로 초기화
                t_cache[zero_indices] = torch.randint(0, FLAGS.T, size=(num_zero_indices,), dtype=torch.long, device=device)
                img_cache[zero_indices] = trainer.diffusion(img_cache[zero_indices],t_cache[zero_indices])



            if step % 100 == 0:  # 예를 들어, 100 스텝마다 시각화
                visualize_t_cache_distribution(t_cache)

            # Logging with WandB
            wandb.log({
                'distill_loss': total_loss.item(),
                'output_loss': output_loss.item()
                       }, step=step)
            pbar.set_postfix(distill_loss='%.3f' % total_loss.item())
             
            # Sample and save student outputs
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                student_model.eval()
                with torch.no_grad():
                    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size).to(device)
                    joint_samples = joint_sampler(x_T)
                    grid = (make_grid(joint_samples, nrow=16) + 1) / 2
                    
                    # Create the directory if it doesn't exist
                    sample_dir = os.path.join(FLAGS.logdir, 'sample')
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    path = os.path.join(sample_dir, 'joint_%d.png' % step)
                    save_image(grid, path)
                    wandb.log({"joint_sample": wandb.Image(path)}, step=step)

                student_model.train()

            # Save student model
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'student_model': student_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'student_ckpt.pt'))

            # Evaluate student model
            if step>0 and FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                student_IS, student_FID, _ = evaluate(student_sampler, student_model)
                metrics = {
                    'Student_IS': student_IS[0],
                    'Student_IS_std': student_IS[1],
                    'Student_FID': student_FID,
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                wandb.log(metrics, step=step)

    wandb.finish()

def compare_models(model1, model2):
    model1_dict = model1.state_dict()
    model2_dict = model2.state_dict()
    
    for (name1, param1), (name2, param2) in zip(model1_dict.items(), model2_dict.items()):
        if param1.size() != param2.size():
            print(f"Layer mismatch: {name1} and {name2}")
            print(f"Size in model 1: {param1.size()}")
            print(f"Size in model 2: {param2.size()}")
        else:
            print(f"Layer {name1} matches in size.")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compare_state_dicts(state_dict1, state_dict2):
    # 두 state_dict의 키가 동일한지 확인
    if state_dict1.keys() != state_dict2.keys():
        print("The models have different parameter keys!")
        return False

    # 각 파라미터 값이 동일한지 확인
    for key in state_dict1.keys():
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        
        if not torch.equal(param1, param2):
            print(f"Parameter mismatch found at: {key}")
            return False
    
    print("All parameters match!")
    return True


def distill_caching_random2():

    # # Initialize WandB
    # wandb.init(
    #     project=FLAGS.wandb_project,
    #     name=FLAGS.wandb_run_name,
    #     notes=FLAGS.wandb_notes,
    #     config={
    #         "learning_rate": FLAGS.lr,
    #         "architecture": "UNet",
    #         "dataset": "your_dataset_name",
    #         "epochs": FLAGS.total_steps,
    #     }
    # )

    pretrained_model_id = "google/ddpm-cifar10-32"
    
    # from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
    # scheduler = DDIMScheduler.from_pretrained("google/ddpm-cifar-256")
    # model = UNet2DModel.from_pretrained("google/ddpm-cifar-256").to("cuda")

    
    
    
    ddpm_pipeline = DDIMPipeline.from_pretrained(pretrained_model_id)
   # 사전 학습된 모델의 설정을 기반으로 커스텀 모델 생성
    unet_config = ddpm_pipeline.unet.config
    
    unet_with_features = UNetWithFeatures(
    sample_size=unet_config.sample_size,
    in_channels=unet_config.in_channels,
    out_channels=unet_config.out_channels,
    layers_per_block=unet_config.layers_per_block,
    block_out_channels=unet_config.block_out_channels,
    norm_num_groups=unet_config.norm_num_groups,
    norm_eps=unet_config.norm_eps,
    time_embedding_type=unet_config.time_embedding_type,
    down_block_types=unet_config.down_block_types,  # down_block_types에서 attention 블록을 포함
    up_block_types=unet_config.up_block_types  # up_block_types에서 attention 블록을 포함
    ).to("cuda")
    
    unet_with_features.load_state_dict(ddpm_pipeline.unet.state_dict(), strict=True)

    print("type(ddpm_pipeline.config): ", type(ddpm_pipeline.unet))
    print("type(unet_with_features): ", type(unet_with_features))

    #teacher_model = UNetWithFeatures.from_pretrained(pretrained_model_id).to(device)


    # # Load pretrained teacher model
    # teacher_model = UNetDiffusers.from_pretrained(pretrained_model_id).to(device)

    # #teacher_model = UNetDiffusers(sample_size=32).to(device)
    # #teacher_model.load_state_dict(pretrained_unet.state_dict())  # 가중치 로드

    # #ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    # #teacher_model.load_state_dict(ckpt['ema_model'])
    # teacher_model.eval()

    # # Initialize student model
    # student_model = UNet(
    #     T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
    #     num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)

    # optim = torch.optim.Adam(student_model.parameters(), lr=FLAGS.lr)
    # sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # trainer = distillation_cache_Trainer(
    #     teacher_model, student_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T,
    #     FLAGS.mean_type, FLAGS.var_type, FLAGS.distill_features).to(device)
    # joint_sampler = GaussianDiffusion_joint_Sampler(
    #     teacher_model, student_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
    #     FLAGS.mean_type, FLAGS.var_type).to(device)
    # student_sampler = GaussianDiffusionSampler(
    #     student_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
    #     FLAGS.mean_type, FLAGS.var_type).to(device)
    
    ddim_pipeline = DDIMPipeline(
    unet=unet_with_features,  # 학습된 UNet 모델
    scheduler=DDPMScheduler()  # DDIM 방식으로 샘플링하기 위한 스케줄러
    ).to("cuda")
    
    generated_images = ddim_pipeline(
    num_inference_steps=100,  # 샘플링 스텝 수
    eta=0.0,                 # deterministic 방식
    batch_size=16             # 한번에 생성할 이미지 배치 크기
    ).images

    import numpy as np
    #print(np.array(generated_images[0]))
    # 3. 생성된 이미지 저장
    for i, image in enumerate(generated_images):
        image.save(f"samples/generated_image_{i}.png")
        
    # ddpm_pipeline의 UNet2DModel 파라미터 개수 출력
    ddpm_unet_params = count_parameters(ddpm_pipeline.unet)
    print(f"ddpm_pipeline UNet2DModel 파라미터 개수: {ddpm_unet_params}")

    # unet_with_features 파라미터 개수 출력
    unet_with_features_params = count_parameters(unet_with_features)
    print(f"unet_with_features 파라미터 개수: {unet_with_features_params}")
    ddpm_pipeline.unet.to("cuda")
    compare_state_dicts(ddpm_pipeline.unet.state_dict(), unet_with_features.state_dict())
    

def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # distill_caching_base()
    distill_caching_random2()


if __name__ == '__main__':
    app.run(main)
