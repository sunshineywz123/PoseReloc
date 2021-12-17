export PYTHONPATH=$PYTHONPATH:$(pwd)
echo $PYTHONPATH
TORCH_USE_RTLD_GLOBAL=YES python3 examples/BundleAdjuster/bundle_adjuster.py --balFile ./data/problem-49-7776-pre.txt --device cuda