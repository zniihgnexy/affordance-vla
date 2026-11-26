"""Crops for the figures of RESEARCH_PROPOSAL.md, cut from tracked images of the weekly reviews.

Every tile is a piece of a figure that a task already produced from real data; nothing is drawn here.

    python scripts/report/proposal_figure_crops.py            # writes report/media/research_proposal/*.png
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "report" / "media"
OUT = MEDIA / "research_proposal"
TILE = 300  # side of a square tile in pixels


def tile(source: str, box: tuple[int, int, int, int], name: str, size: tuple[int, int] = (TILE, TILE)) -> None:
    """Cut `box` (left, top, right, bottom) out of `source` and save it at `size`."""
    image = Image.open(MEDIA / source).convert("RGB").crop(box).resize(size, Image.LANCZOS)
    image.save(OUT / name)


def side_by_side(source: str, boxes: list[tuple[int, int, int, int]], name: str, gap: int = 6) -> None:
    """Cut several boxes out of one image and put them in a row on white."""
    parts = [Image.open(MEDIA / source).convert("RGB").crop(box).resize((TILE, TILE), Image.LANCZOS) for box in boxes]
    canvas = Image.new("RGB", (len(parts) * TILE + (len(parts) - 1) * gap, TILE), "white")
    for index, part in enumerate(parts):
        canvas.paste(part, (index * (TILE + gap), 0))
    canvas.save(OUT / name)


def stacked(source: str, boxes: list[tuple[int, int, int, int]], name: str, gap: int = 6) -> None:
    """Cut several equally wide boxes out of one image and stack them on white."""
    parts = [Image.open(MEDIA / source).convert("RGB").crop(box) for box in boxes]
    width = max(part.width for part in parts)
    canvas = Image.new("RGB", (width, sum(part.height for part in parts) + (len(parts) - 1) * gap), "white")
    top = 0
    for part in parts:
        canvas.paste(part, ((width - part.width) // 2, top))
        top += part.height + gap
    canvas.save(OUT / name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    replay = "weekly_2026_11_23/robocasa365/replay_video_vs_rerender.png"  # AFFO-10: CoffeeSetupMug, episode 0
    side_by_side(replay, [(386, 28, 642, 284), (898, 28, 1154, 284), (1154, 28, 1410, 284)], "replay_rgb_depth_seg.png")
    # AFFO-9: the red mug's own demonstrated grasps (orange) on one demonstration frame
    tile("weekly_2026_11_23/mug_transfer/red_coffee_mug.png", (893, 313, 1023, 443), "demo_grasps_on_mug.png")
    # AFFO-10: two RoboCasa365 mugs as surface points, one of the training split and one held out (side and top view)
    stacked(
        "weekly_2026_11_23/robocasa365/category_surface_points.png",
        [(150, 0, 450, 165), (750, 0, 1050, 165)],
        "category_instances.png",
    )
    # AFFO-11: touch frequency and closing axes (cyan) on a scanned mug, from human-hand data
    tile("weekly_2026_11_23/hot3d/contact.png", (88, 135, 258, 305), "map_on_surface.png")
    for name in ("input", "map_now", "map_v3", "plan", "counterfactual"):  # frames of a real LIBERO-PRO rollout
        tile(f"pipeline_overview/{name}.png", (0, 0, 600, 600), f"frame_{name}.png")


if __name__ == "__main__":
    main()
