import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle
from matplotlib.animation import FuncAnimation

# Initialize figure and axis
fig, ax = plt.subplots()

# Define the points
points = np.array([[4.5, 2], [3, 4], [5, 6]])

# Plot the points
point1, = ax.plot(points[:, 0], points[:, 1], 'bo')  # Point 1
line1, = ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], 'r-')  # Line 1
line2, = ax.plot([points[1][0], points[2][0]], [points[1][1], points[2][1]], 'r-')  # Line 2

# Define the circle
circle_center = (4, 8)
circle_radius = 2

# Plot the circle
circle = Circle(circle_center, circle_radius, edgecolor='b', facecolor='none')
ax.add_patch(circle)

# Initialize arc
arc_center = points[1]
arc_radius = 1.5
start_angle = np.arctan2(points[0][1] - arc_center[1], points[0][0] - arc_center[0])
start_angle_deg = np.degrees(start_angle)
if start_angle_deg < 0:
    start_angle_deg += 360
arc = Arc(arc_center, arc_radius*2, arc_radius*2, angle=start_angle_deg, theta1=0, theta2=0, color='k')
ax.add_patch(arc)

# # Animation function
# def animate(frame):
#     # Update point 2 position
#     points[1][0] = 3 + frame / 10  # Move point 2 from left to right
    
#     # Update the plot elements
#     point1.set_data(points[:, 0], points[:, 1])
#     line1.set_data([points[0][0], points[1][0]], [points[0][1], points[1][1]])
#     line2.set_data([points[1][0], points[2][0]], [points[1][1], points[2][1]])
    
#     # Update the angle arc
#     arc.theta2 = 180 - np.degrees(np.arccos(np.dot(points[1] - points[0], points[2] - points[1]) / (np.linalg.norm(points[1] - points[0]) * np.linalg.norm(points[2] - points[1])))) * (frame + 1) / 100
    
#     return point1, line1, line2

# # Create animation
# ani = FuncAnimation(fig, animate, frames=100, interval=10, blit=True)


# Animation function
def animate(frame):
    print(frame)
    # Update point 2 position
    points[1][0] = 3 + frame / 10  # Move point 2 from left to right

    print( points[1][0] )
    
    # Update the plot elements
    point1.set_data(points[:, 0], points[:, 1])
    line1.set_data([points[0][0], points[1][0]], [points[0][1], points[1][1]])
    line2.set_data([points[1][0], points[2][0]], [points[1][1], points[2][1]])
    
    # Calculate the angle between the lines
    vector1 = points[1] - points[0]
    vector2 = points[2] - points[1]
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # Update the angle arc
    arc.theta2 = 180 - angle_deg * (frame + 1) / 100
    
    # Update the angle annotation
    angle_annotation.set_text(f'{angle_deg:.2f}°')
    angle_annotation.set_position((points[1][0] + 0.5, points[1][1] + 0.5))
    
    return point1, line1, line2, angle_annotation

# Create the angle annotation
angle_annotation = ax.text(points[1][0] + 0.5, points[1][1] + 0.5, '', fontsize=12)

# Create animation
ani = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)


# Set plot limits
plt.xlim(0, 12)
plt.ylim(0, 8)

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Animation: Moving Point 2 from Left to Right')

# Show the animation
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
