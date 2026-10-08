"""Screenshot annotations drawn from observed geometry, without modifying the page."""

from PIL import ImageDraw, ImageFont


def annotate(image, snapshot, selected_node=None):
    """Draw one labelled box per observed DOM node; orange marks the chosen target.

    Args:
        image: Screenshot matching the snapshot viewport.
        snapshot: Observed actions with node IDs and viewport-relative rectangles.
        selected_node: Chosen DOM node to highlight before execution, if any.

    Returns:
        A copy of the screenshot, with display-only annotations.
    """
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=14)
    sx, sy = image.width / snapshot["w"], image.height / snapshot["h"]
    nodes = {
        a["node"]: a for a in snapshot["actions"] if a.get("rect") and a.get("node") is not None
    }
    # Draw the selected target last so overlapping outlines cannot obscure it.
    for node, action in sorted(nodes.items(), key=lambda item: item[0] == selected_node):
        rect = action["rect"]
        x, y = max(0, rect["x"] * sx), max(0, rect["y"] * sy)
        right, bottom = (
            min(out.width - 1, (rect["x"] + rect["w"]) * sx),
            min(out.height - 1, (rect["y"] + rect["h"]) * sy),
        )
        if right <= x or bottom <= y:
            continue
        color = "#f97316" if node == selected_node else "#2563eb"
        draw.rectangle(
            (x, y, right, bottom), outline=color, width=4 if node == selected_node else 2
        )
        label = str(node)
        box = draw.textbbox((0, 0), label, font=font)
        width, height = box[2] - box[0] + 8, box[3] - box[1] + 6
        lx, ly = min(x, max(0, out.width - width)), max(0, y - height)
        draw.rectangle((lx, ly, lx + width, ly + height), fill=color)
        draw.text((lx + 4, ly + 3 - box[1]), label, font=font, fill="white")
    return out
