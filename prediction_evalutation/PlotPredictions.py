"""
This script is a helper script to visualize
"""
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import Range1d


def plot_pred_vs_gt(ground_truth: np.array, prediction: np.array):
    # Define colors for exons, introns, and intergenic lines
    colors = {0: "lightgreen", 2: "lightblue", 8: "black"}

    # Efficient numpy-based function to group consecutive identical elements
    def group_annotation(annotation):
        # Find changes between consecutive elements
        changes = np.diff(annotation, prepend=np.nan)
        group_boundaries = np.where(changes != 0)[0]

        # Create tuples for (region_type, start, end)
        return [(annotation[start], start, end) for start, end in zip(group_boundaries, np.append(group_boundaries[1:], len(annotation)))]

    # Group annotations for ground truth and prediction using numpy
    grouped_ground_truth = group_annotation(ground_truth)
    grouped_prediction = group_annotation(prediction)

    # Create Bokeh figure with horizontal tracks
    p = figure(height=800, width=1700, title="Genome Annotation (Ground Truth vs Prediction)",
               tools="pan,wheel_zoom,box_zoom,reset", x_axis_label='Position', y_axis_label='Tracks')

    # Adjust track positions to bring them closer together
    ground_truth_track_pos = 0.3
    prediction_track_pos = 0.2
    track_height = 0.05  # Reduced height of the tracks

    # Ground truth track (exons/introns)
    for region_type, start, end in grouped_ground_truth:
        if region_type == 8:
            # Plot intergenic regions as thin lines, aligned with the tracks
            p.line(x=[start, end], y=[ground_truth_track_pos, ground_truth_track_pos],
                   line_width=1, color=colors[region_type], alpha=0.6, legend_label="Intergenic")
        else:
            p.quad(top=[ground_truth_track_pos + track_height], bottom=[ground_truth_track_pos - track_height],
                   left=[start], right=[end], color=colors[region_type], alpha=0.6,
                   legend_label="Exon" if region_type == 0 else "Intron")

    # Prediction track (exons/introns)
    for region_type, start, end in grouped_prediction:
        if region_type == 8:
            # Plot intergenic regions as thin lines, aligned with the tracks
            p.line(x=[start, end], y=[prediction_track_pos, prediction_track_pos],
                   line_width=1, color=colors[region_type], alpha=0.6, legend_label="Intergenic")
        else:
            p.quad(top=[prediction_track_pos + track_height], bottom=[prediction_track_pos - track_height],
                   left=[start], right=[end], color=colors[region_type], alpha=0.6,
                   legend_label="Exon" if region_type == 0 else "Intron")

    # Set y-axis range and labels for better readability
    p.y_range = Range1d(-0.5, 1)
    p.yaxis.ticker = [prediction_track_pos, ground_truth_track_pos]
    p.yaxis.major_label_overrides = {prediction_track_pos: "Prediction", ground_truth_track_pos: "Ground Truth"}

    # Customize legend to show unique entries
    p.legend.location = "top_left"
    p.legend.title = "Annotations"
    p.legend.label_text_font_size = '8pt'  # Smaller legend text size
    p.legend.click_policy = "hide"  # Allow hiding of lines/rectangles via legend

    # Enable output to HTML and show plot in the browser
    output_file("bokeh_genome_tracks.html")
    show(p)
