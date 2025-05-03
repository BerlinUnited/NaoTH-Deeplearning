import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# Define the soccer field dimensions
field_length = 9  # meters
field_width = 6   # meters

# Define the robot's position and orientation
robot_position = np.array([5, 5])  # (x, y) in meters
robot_angle = 135  # degrees, facing direction
fov_angle = 45  # field of view in degrees
view_distance = 2  # meters

# Define the lines on the soccer field (start and end points)
field_lines = [
    ((0, 0), (field_length, 0)),  # Bottom boundary
    ((0, field_width), (field_length, field_width)),  # Top boundary
    ((0, 0), (0, field_width)),  # Left boundary
    ((field_length, 0), (field_length, field_width)),  # Right boundary
    ((field_length / 2, 0), (field_length / 2, field_width)),  # Center line
    ((0, field_width / 4), (field_length / 4, field_width / 4)),  # Example penalty line
    ((field_length * 3 / 4, field_width / 4), (field_length, field_width / 4)),  # Example penalty line
]

# Function to calculate the intersection points between a line and the robot's FOV
def line_segment_in_fov(line, robot_pos, angle, fov_angle, view_distance):
    start, end = np.array(line[0]), np.array(line[1])
    robot_pos = np.array(robot_pos)

    # Convert angles to radians
    angle_rad = np.radians(angle)
    fov_half_rad = np.radians(fov_angle) / 2

    # Calculate the direction vectors of the FOV boundaries
    dir1 = np.array([np.cos(angle_rad - fov_half_rad), np.sin(angle_rad - fov_half_rad)])
    dir2 = np.array([np.cos(angle_rad + fov_half_rad), np.sin(angle_rad + fov_half_rad)])

    # Parametric representation of the line segment
    line_vec = end - start
    line_length = np.linalg.norm(line_vec)
    line_dir = line_vec / line_length

    # Find intersection points with the FOV boundaries
    intersections = []
    for d in [dir1, dir2]:
        denominator = d[0] * line_dir[1] - d[1] * line_dir[0]
        if np.abs(denominator) < 1e-6:
            continue  # Parallel lines, no intersection
        t = ((robot_pos[0] - start[0]) * d[1] - (robot_pos[1] - start[1]) * d[0]) / denominator
        if 0 <= t <= line_length:
            intersection = start + t * line_dir
            intersections.append(intersection)

    # Find intersection points with the circular boundary of the FOV
    # Parametric representation of the line: start + t * line_dir
    # Distance from robot_pos to the line: |(start - robot_pos) × line_dir| / |line_dir|
    distance = np.abs(np.cross(start - robot_pos, line_dir)) / np.linalg.norm(line_dir)
    if distance <= view_distance:
        # Find the points where the line intersects the circle
        a = np.dot(line_dir, line_dir)
        b = 2 * np.dot(line_dir, start - robot_pos)
        c = np.dot(start - robot_pos, start - robot_pos) - view_distance**2
        discriminant = b**2 - 4 * a * c
        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)
            for t in [t1, t2]:
                if 0 <= t <= line_length:
                    intersection = start + t * line_dir
                    intersections.append(intersection)

    # Filter intersections to only those within the FOV angle
    valid_intersections = []
    for point in intersections:
        vec = point - robot_pos
        angle_to_point = np.degrees(np.arctan2(vec[1], vec[0]))
        angle_diff = (angle_to_point - angle + 180) % 360 - 180
        if -fov_angle / 2 <= angle_diff <= fov_angle / 2:
            valid_intersections.append(point)

    # If there are two valid intersections, return the segment between them
    if len(valid_intersections) == 2:
        return valid_intersections
    else:
        return []

# Plot the soccer field
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, field_length)
ax.set_ylim(0, field_width)
ax.set_aspect('equal')

# Draw the field lines
for line in field_lines:
    ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color='black')

# Draw the robot's position
ax.plot(robot_position[0], robot_position[1], 'ro', label='Robot')

# Draw the robot's field of view
fov_wedge = Wedge(robot_position, view_distance, robot_angle - fov_angle / 2, robot_angle + fov_angle / 2, color='blue', alpha=0.2)
ax.add_patch(fov_wedge)

# Highlight the segments of the lines that are in view
for line in field_lines:
    intersections = line_segment_in_fov(line, robot_position, robot_angle, fov_angle, view_distance)
    if len(intersections) == 2:
        ax.plot([intersections[0][0], intersections[1][0]], [intersections[0][1], intersections[1][1]], color='red', linewidth=2)
    else:
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color='gray', linestyle='--')

# Add labels and legend
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_title('Robot Field of View Simulation')
ax.legend()

plt.show()
