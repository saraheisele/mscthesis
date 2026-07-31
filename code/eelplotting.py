"""SVG eel drawing helpers for position animation.

Used only by ``position_estimation/animate_eel_position.py``.
Not part of the main ``run_analysis.sh`` pipeline.
"""

import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from svgpath2mpl import parse_path


def load_svg_paths(svg_file_path):
    """Return path ``d`` attribute strings from an SVG file."""
    tree = ET.parse(svg_file_path)
    root = tree.getroot()

    namespace = {"svg": "http://www.w3.org/2000/svg"}
    path_elements = root.findall(".//svg:path", namespace)

    if not path_elements:
        raise ValueError("No path elements found in the SVG file.")

    svg_paths_data = [element.attrib["d"] for element in path_elements]
    return svg_paths_data


def extract_vertices(svg_file_path):
    """Parse SVG paths into matplotlib vertex arrays (one array per path)."""
    svg_paths_data = load_svg_paths(svg_file_path)

    all_verts = []
    for svg_path_data in svg_paths_data:
        path = parse_path(svg_path_data)
        verts = path.vertices
        all_verts.append(verts)

    return all_verts


def normalize_vertices(verts):
    """Normalize body/fin vertices to unit width while preserving aspect ratio."""
    min_x, min_y = np.min(verts[0], axis=0)
    max_x, max_y = np.max(verts[0], axis=0)

    width = max_x - min_x
    height = max_y - min_y
    aspect_ratio = height / width

    body = verts[0].copy()
    body[:, 0] = (body[:, 0] - min_x) / width
    body[:, 1] = (body[:, 1] - min_y) / height * aspect_ratio

    fin = verts[1].copy()
    fin[:, 0] = (fin[:, 0] - min_x) / width
    fin[:, 1] = (fin[:, 1] - min_y) / height * aspect_ratio

    normalized_verts = [body, fin]

    return normalized_verts


def add_curvature(verts, x, y):
    """Warp body/fin outlines onto a centre-line curve ``(x, y)``."""
    body = verts[0]
    fin = verts[1]

    # Scale fish outline to the centre-line length.
    abs_x_range = np.max(x) - np.min(x)
    verts_x_range = np.max(body[:, 0]) - np.min(body[:, 0])
    scale_factor = abs_x_range / verts_x_range

    fin = fin * scale_factor
    body = body * scale_factor

    fin[:, 0] = fin[:, 0] - np.max(body[:, 0])
    body[:, 0] = body[:, 0] - np.max(body[:, 0])

    body = body[body[:, 0].argsort()]

    x = x[x.argsort()]
    y = y[x.argsort()]

    # Interpolate outline thickness onto the centre-line samples.
    new_y = np.interp(x, body[:, 0], body[:, 1])

    fin_centroid_x = np.mean(fin[:, 0])
    fin_centroid_y = np.mean(fin[:, 1])

    closest_fin_idx = np.argmin(np.abs(x - fin_centroid_x))

    dx = np.gradient(x)
    dy = np.gradient(y)
    normal = np.array([-dy, dx]).T

    normal /= np.linalg.norm(normal, axis=1)[:, np.newaxis]

    new_body = np.array([x, y]).T + normal * new_y[:, np.newaxis]

    centered_fin = fin - np.array([fin_centroid_x, fin_centroid_y])

    angle = np.arctan2(normal[closest_fin_idx, 1], normal[closest_fin_idx, 0])
    angle = np.pi / 2 - angle

    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    new_fin = np.dot(centered_fin, rotation_matrix)

    fin_pos = (
        np.array([x[closest_fin_idx], y[closest_fin_idx]]).T
        + normal[closest_fin_idx] * fin_centroid_y
    )
    moved_fin = new_fin + fin_pos

    return [new_body, moved_fin]


def mirror_vertices(verts, axis="y"):
    """Mirror vertex arrays across the given axis (``'x'`` or ``'y'``)."""
    if axis == "y":
        mirrored_verts = [v * np.array([-1, 1]) for v in verts]
    elif axis == "x":
        mirrored_verts = [v * np.array([1, -1]) for v in verts]
    else:
        raise ValueError("Axis should be 'x' or 'y'.")

    return mirrored_verts


def plot_eel(ax, original_verts, mirrored_verts, **kwargs):
    """Fill body and fins on ``ax``; returns the created artists."""
    body = np.vstack((original_verts[0], mirrored_verts[0][::-1]))
    fin1 = original_verts[1]
    fin2 = mirrored_verts[1]

    artists = []
    artists.extend(ax.fill(fin1[:, 0], fin1[:, 1], zorder=5, **kwargs))
    artists.extend(ax.fill(fin2[:, 0], fin2[:, 1], zorder=5, **kwargs))
    artists.extend(ax.fill(body[:, 0], body[:, 1], zorder=15, **kwargs))
    return artists


def get_eel_shape(svgfile, x, y, eellen, rotate=0, headpos=(0, 0)):
    """Build curved eel outlines from an SVG, centre line, length, and pose."""
    original_verts = extract_vertices(svgfile)

    original_verts = mirror_vertices(original_verts, axis="x")
    original_verts = normalize_vertices(original_verts)
    mirrored_verts = mirror_vertices(original_verts, axis="x")

    original_verts = add_curvature(original_verts, x, y)
    mirrored_verts = add_curvature(mirrored_verts, x, y)

    rotation_matrix = np.array(
        [
            [np.cos(np.radians(rotate)), -np.sin(np.radians(rotate))],
            [np.sin(np.radians(rotate)), np.cos(np.radians(rotate))],
        ]
    )
    original_verts = [np.dot(v, rotation_matrix) for v in original_verts]
    mirrored_verts = [np.dot(v, rotation_matrix) for v in mirrored_verts]
    x, y = np.dot(np.array([x, y]).T, rotation_matrix).T

    current_headpos = original_verts[0][-1]
    original_verts = [v - current_headpos for v in original_verts]
    x = x - current_headpos[0]
    y = y - current_headpos[1]

    original_verts = [v + np.array(headpos) for v in original_verts]
    mirrored_verts = [v + np.array(headpos) for v in mirrored_verts]
    x = x + headpos[0]
    y = y + headpos[1]

    return [original_verts, mirrored_verts], x, y


if __name__ == "__main__":
    svg_file = "assets/eel.svg"
    eellen = 200
    fig, ax = plt.subplots()

    colors = sns.color_palette("husl", 5)
    neels = len(colors)
    angles = np.linspace(0, 360, neels, endpoint=False)[::-1]

    # Centre line for the fish body. add_curvature/get_eel_shape place the
    # head at x=0 and lay the body out toward negative x, and the body
    # thickness is sampled onto this centre line with np.interp -- so the
    # centre line must live in the same [-eellen, 0] domain, otherwise the
    # thickness gets clamped to ~0 and the body collapses to a flat line.
    x = np.linspace(-eellen, 0, 1000)
    y = np.zeros_like(x)
    eel, x, y = get_eel_shape(svg_file, x, y, eellen)
    plot_eel(ax, eel[0], eel[1], color="k")

    ax.set_aspect("equal")
    plt.show()
