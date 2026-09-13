# Upload RunwayGuard weights to Hugging Face (PowerShell)
# 1) Create a free account at https://huggingface.co and a Write token
# 2) Run:  pip install -U "huggingface_hub[cli]"
# 3) Run:  hf auth login
# 4) Edit $User below, then run this script from the RunwayGuard folder

$User = "DarkKnight1217"   # e.g. riyaz-ahamed-07
$Repo = "$User/RunwayGuard-rtdetr-l"

hf repo create RunwayGuard-rtdetr-l --type model 2>$null
hf upload $Repo hf_upload . --repo-type=model --commit-message "Add RunwayGuard RT-DETR-L best.pt + metrics"

Write-Host "Public URL: https://huggingface.co/$Repo"
Write-Host "Direct file: https://huggingface.co/$Repo/resolve/main/best.pt"
Write-Host "Then replace DarkKnight1217 in README.md, MEMO.md, docs/hf/MODEL_CARD.md"
