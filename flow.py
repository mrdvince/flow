from pathlib import Path

import fire
import torchvision
import torchvision.transforms.functional as F
from torchvision.io import write_jpeg
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import flow_to_image

from stitch import write_video


def runner(video, to=None, ss=0, filename="dummy.mp4"):
    video = Path(video)
    vframes, aframes, metadata = torchvision.io.read_video(
        str(video), pts_unit="sec", output_format="TCHW", start_pts=ss, end_pts=to
    )
    model, transforms = flow_model()
    dest_dir = Path("flows")
    dest_dir.mkdir(exist_ok=True, parents=True)
    for idx, (img1, img2) in enumerate(zip(vframes, vframes[1:])):
        img1, img2 = preprocess(
            img1=img1.unsqueeze(0), img2=img2.unsqueeze(0), transforms=transforms
        )
        flows = model(img1.to(device), img2.to(device))
        pred_flow = flows[-1][0]
        flow_img = flow_to_image(pred_flow).to("cpu")
        write_jpeg(flow_img, str(dest_dir / f"predicted_flow_{idx}.jpg"))
    write_video(dest_dir, filename, aframes, metadata)


def flow_model():
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device)
    model = model.eval()
    transforms = weights.transforms()
    return model, transforms


def preprocess(img1, img2, transforms):
    img1 = F.resize(img1, size=[520, 960])
    img2 = F.resize(img2, size=[520, 960])
    return transforms(img1, img2)


if __name__ == "__main__":
    # os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    fire.Fire(runner)
# ffmpeg -i right.mp4 -vf "fps=24,scale=960:520:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" right.gif
