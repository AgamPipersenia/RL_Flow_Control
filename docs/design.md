Terminal Usage --

git init
git add .
git config --global user.email {Email}
git config --global useer.name {User Name}
git commit -m "Initial Project Setup"
git branch -M main
git remote add origin {Project URL}
git push -u origin main

GitHub removed support for password authentication over HTTPS a while ago. Now, you need to use one of these methods instead:

ls ~/.ssh {Look for files like id_rsa and id_rsa.pub.}
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub

ssh -T git@github.com

sign_and_send_pubkey: signing failed for ED25519 "/home/monarch/.ssh/id_ed25519" from agent: agent refused operation
git@github.com: Permission denied (publickey).

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

ssh -T git@github.com

Hi Aman-Pandey923! You've successfully authenticated, but GitHub does not provide shell access.
fatal: Authentication failed 

Switch your Git remote from HTTPS ➜ SSH

git remote set-url origin git@github.com:{xyz}.git
git remote -v (to confirm)
git push -u origin main

PyTorch: Check if GPU is available

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

python --version
pip install --upgrade numpy matplotlib stable-baselines3 gymnasium
pip cache purge
pip install --upgrade numpy matplotlib
pip install --upgrade stable-baselines3
pip install --upgrade gymnasium
pip install phiflow
pip install dash
pip install jax
pip install tqdm

nvidia-smi: shows the CUDA version at the top right.
pip uninstall jax jaxlib -y

Install JAX with CUDA 12.8 Support --
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Verify GPU Usage in JAX --
import jax
print(jax.devices())
