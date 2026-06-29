import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from svgpath2mpl import parse_path
import numpy as np

# from eeltracker.efield import fish_coords
import seaborn as sns


def load_svg_paths(svg_file_path):
    # Parse the SVG file
    tree = ET.parse(svg_file_path)
    root = tree.getroot()

    # Find all 'path' elements
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    path_elements = root.findall(".//svg:path", namespace)

    if not path_elements:
        raise ValueError("No path elements found in the SVG file.")

    # Extract the 'd' attribute (the path data) from each path element
    svg_paths_data = [element.attrib["d"] for element in path_elements]
    return svg_paths_data


def extract_vertices(svg_file_path):
    svg_paths_data = load_svg_paths(svg_file_path)

    # Parse the path and extract vertices
    all_verts = []
    for svg_path_data in svg_paths_data:
        path = parse_path(svg_path_data)
        verts = path.vertices  # Extract the vertices (points) from the path
        all_verts.append(verts)

    return all_verts  # Combine all vertices into a single array


def normalize_vertices(verts):
    # Get the minimum and maximum values of the vertices
    min_x, min_y = np.min(verts[0], axis=0)
    max_x, max_y = np.max(verts[0], axis=0)

    # Calculate the width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y
    aspect_ratio = height / width

    # Normalize the x coordinates to be between 0 and 1
    # but keep the aspect ratio of the original shape
    body = verts[0].copy()
    body[:, 0] = (body[:, 0] - min_x) / width
    body[:, 1] = (body[:, 1] - min_y) / height * aspect_ratio

    fin = verts[1].copy()
    fin[:, 0] = (fin[:, 0] - min_x) / width
    fin[:, 1] = (fin[:, 1] - min_y) / height * aspect_ratio

    normalized_verts = [body, fin]

    return normalized_verts


def add_curvature(verts, x, y):
    # split body and fins
    body = verts[0]
    fin = verts[1]

    # scale up the fish to match the center line
    abs_x_range = np.max(x) - np.min(x)
    verts_x_range = np.max(body[:, 0]) - np.min(body[:, 0])
    scale_factor = abs_x_range / verts_x_range

    fin = fin * scale_factor
    body = body * scale_factor

    fin[:, 0] = fin[:, 0] - np.max(body[:, 0])
    body[:, 0] = body[:, 0] - np.max(body[:, 0])

    # sort verts by x
    body = body[body[:, 0].argsort()]

    # sort x and y by x
    x = x[x.argsort()]
    y = y[x.argsort()]

    # Linearly interpolate the y values of the vertices
    # to match the number of points in the center line
    new_y = np.interp(x, body[:, 0], body[:, 1])

    # Compute the centroid of the fin
    fin_centroid_x = np.mean(fin[:, 0])
    fin_centroid_y = np.mean(fin[:, 1])

    # Find the index on the body closest to the centroid of the fin
    closest_fin_idx = np.argmin(np.abs(x - fin_centroid_x))

    # For each point on the center line, find the normal vector
    # i.e. the one orthogonal to the tangent vector
    dx = np.gradient(x)
    dy = np.gradient(y)
    normal = np.array([-dy, dx]).T

    # make normal unit vectors
    normal /= np.linalg.norm(normal, axis=1)[:, np.newaxis]

    # Move each vertex along the normal vector
    # with the distance given by the interpolated y value
    new_body = np.array([x, y]).T + normal * new_y[:, np.newaxis]

    # Rotate the fin by the angle between the normal vector
    # at the closest point on the body and the x-axis
    centered_fin = fin - np.array([fin_centroid_x, fin_centroid_y])

    # Get the angle between the normal vector and the x-axis
    # where the fin is closest to the body
    angle = np.arctan2(normal[closest_fin_idx, 1], normal[closest_fin_idx, 0])

    # Difference between 90 deg and the angle
    angle = np.pi / 2 - angle

    # Rotate the fin
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    new_fin = np.dot(centered_fin, rotation_matrix)

    # Get the new fin pos from normal vector and original centroid
    fin_pos = (
        np.array([x[closest_fin_idx], y[closest_fin_idx]]).T
        + normal[closest_fin_idx] * fin_centroid_y
    )
    moved_fin = new_fin + fin_pos

    # # Move the fin to the new position along the normal vector
    # plt.close()
    #
    # # Plot the center line
    # plt.plot(x, y, "k--", lw=1)
    #
    # # Plot the interpolated y values
    # plt.plot(x, new_y, "r-", lw=2)
    #
    # # Plot the original vertices
    # plt.plot(body[:, 0], body[:, 1], "b-", lw=2)
    # plt.plot(fin[:, 0], fin[:, 1], "b-", lw=2)
    #
    # # Plot the new fin position
    # plt.plot(moved_fin[:, 0], moved_fin[:, 1], "r-", lw=2)
    #
    # # plot every tenth normal vector
    # for i in range(0, len(x), 100):
    #     plt.quiver(x[i], y[i], normal[i, 0], normal[i, 1], scale=10, color="r")
    #
    # plt.quiver(
    #     x[closest_fin_idx],
    #     y[closest_fin_idx],
    #     normal[closest_fin_idx, 0],
    #     normal[closest_fin_idx, 1],
    #     scale=10,
    #     color="g",
    # )
    #
    # # Plot the new vertices
    # plt.plot(new_body[:, 0], new_body[:, 1], "r-", lw=2)
    #
    # # equal aspect ratio
    # plt.gca().set_aspect("equal", adjustable="box")
    # plt.show()

    return [new_body, moved_fin]


def mirror_vertices(verts, axis="y"):
    if axis == "y":
        mirrored_verts = [v * np.array([-1, 1]) for v in verts]
    elif axis == "x":
        mirrored_verts = [v * np.array([1, -1]) for v in verts]
    else:
        raise ValueError("Axis should be 'x' or 'y'.")

    return mirrored_verts


def plot_eel(ax, original_verts, mirrored_verts, **kwargs):
    body = np.vstack((original_verts[0], mirrored_verts[0][::-1]))
    fin1 = original_verts[1]
    fin2 = mirrored_verts[1]

    artists = []
    artists.extend(ax.fill(fin1[:, 0], fin1[:, 1], zorder=5, **kwargs))
    artists.extend(ax.fill(fin2[:, 0], fin2[:, 1], zorder=5, **kwargs))
    artists.extend(ax.fill(body[:, 0], body[:, 1], zorder=15, **kwargs))
    return artists


def get_eel_shape(svgfile, x, y, eellen, rotate=0, headpos=(0, 0)):
    # Extract the vertices from the SVG
    original_verts = extract_vertices(svgfile)

    # Mirror the vertices along the x-axis to get the other half of the fish
    original_verts = mirror_vertices(original_verts, axis="x")

    # Normalize the vertices
    original_verts = normalize_vertices(original_verts)

    # Mirror the vertices along the y-axis to get the other half of the fish
    mirrored_verts = mirror_vertices(original_verts, axis="x")

    # Transform the vertices to match the center line
    original_verts = add_curvature(original_verts, x, y)
    mirrored_verts = add_curvature(mirrored_verts, x, y)

    # Rotate the fish
    rotation_matrix = np.array(
        [
            [np.cos(np.radians(rotate)), -np.sin(np.radians(rotate))],
            [np.sin(np.radians(rotate)), np.cos(np.radians(rotate))],
        ]
    )
    original_verts = [np.dot(v, rotation_matrix) for v in original_verts]
    mirrored_verts = [np.dot(v, rotation_matrix) for v in mirrored_verts]
    x, y = np.dot(np.array([x, y]).T, rotation_matrix).T

    # Move head back to (0, 0)
    current_headpos = original_verts[0][-1]
    original_verts = [v - current_headpos for v in original_verts]
    x = x - current_headpos[0]
    y = y - current_headpos[1]

    # Position the fish
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
    # x, y, _ = fish_coords(cp1=cp1, cp2=cp2, eellen=eellen)
    x = np.linspace(-eellen, 0, 1000)
    y = np.zeros_like(x)
    eel, x, y = get_eel_shape(svg_file, x, y, eellen)
    plot_eel(ax, eel[0], eel[1], color="k")

    ax.set_aspect("equal")
    plt.show()
