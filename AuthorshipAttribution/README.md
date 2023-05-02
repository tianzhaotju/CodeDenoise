### Preparing the Dataset
--- --- ---
***The datasets are already prepared in the container, and you can skip this step.***

```shell
cd /root/CodeDenoise/AuthorshipAttribution/code/;

python gen_var.py;

python gen_data.py --model_name=codebert;
python gen_data.py --model_name=graphcodebert;
python gen_data.py --model_name=codet5;
```

### Input Denoising

```shell
cd /root/CodeDenoise/AuthorshipAttribution/code/;

CUDA_VISIBLE_DEVICES=0 python denoise.py \
    --model_name=codebert --theta=1 --N=1;
CUDA_VISIBLE_DEVICES=0 python denoise.py \
    --model_name=graphcodebert --theta=1 --N=1;
CUDA_VISIBLE_DEVICES=0 python denoise.py \
    --model_name=codet5 --theta=1 --N=1;
```