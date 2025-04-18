import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import Range1d # Import Range1d

# Helper function to group consecutive identical elements (remains the same)
def group_annotation(annotation):
    """Efficiently groups consecutive identical elements in a numpy array."""
    # Find changes between consecutive elements
    changes = np.diff(annotation, prepend=np.nan)
    group_boundaries = np.where(changes != 0)[0]

    # Create tuples for (region_type, start, end)
    return [
        (annotation[start], start, end)
        for start, end in zip(group_boundaries, np.append(group_boundaries[1:], len(annotation)))
    ]

def plot_pred_vs_gt(
    ground_truth: np.array,
    prediction: np.array,
    reading_frame: np.array = None # Optional reading frame array
):
    """
    Plots genome annotation tracks for ground truth, prediction, and optionally reading frames.

    Args:
        ground_truth (np.array): Array representing ground truth annotations.
                                 Expected values: 0 (exon), 2 (intron), 8 (intergenic).
        prediction (np.array): Array representing predicted annotations.
                               Expected values: 0 (exon), 2 (intron), 8 (intergenic).
        reading_frame (np.array, optional): Array representing reading frames.
                                             Expected values: 0, 1, 2 for frames,
                                             np.inf for non-coding/no match.
                                             Defaults to None (track not plotted).
    """
    # Define colors for exons, introns, intergenic, and reading frames
    anno_colors = {0: "lightgreen", 2: "lightblue", 8: "black"}
    rf_colors = {0: "coral", 1: "skyblue", 2: "lightsalmon"} # Colors for frames 0, 1, 2

    # Group annotations for ground truth and prediction using numpy
    grouped_ground_truth = group_annotation(ground_truth)
    grouped_prediction = group_annotation(prediction)
    grouped_reading_frame = None
    if reading_frame is not None:
        # Ensure reading_frame is a numpy array for processing
        if not isinstance(reading_frame, np.ndarray):
             reading_frame = np.array(reading_frame)
        grouped_reading_frame = group_annotation(reading_frame)

    # Create Bokeh figure
    p = figure(
        height=800,
        width=1700,
        title="Genome Annotation Comparison",
        tools="pan,wheel_zoom,box_zoom,reset,save", # Added save tool
        x_axis_label="Position",
        y_axis_label="Tracks",
    )

    # Define track positions and height
    ground_truth_track_pos = 0.6
    prediction_track_pos = 0.4
    reading_frame_track_pos = 0.2 # Position for the new track
    track_height = 0.06 # Adjusted height for potentially 3 tracks

    # --- Plot Ground Truth Track ---
    for region_type, start, end in grouped_ground_truth:
        legend_label = "GT " + ("Exon" if region_type == 0 else "Intron" if region_type == 2 else "Intergenic")
        if region_type == 8: # Intergenic
            p.line(
                x=[start, end],
                y=[ground_truth_track_pos, ground_truth_track_pos],
                line_width=1,
                color=anno_colors[region_type],
                alpha=0.6,
                legend_label=legend_label,
            )
        elif region_type in anno_colors: # Exon or Intron
            p.quad(
                top=[ground_truth_track_pos + track_height],
                bottom=[ground_truth_track_pos - track_height],
                left=[start],
                right=[end],
                color=anno_colors[region_type],
                alpha=0.6,
                legend_label=legend_label,
            )

    # --- Plot Prediction Track ---
    for region_type, start, end in grouped_prediction:
        legend_label = "Pred " + ("Exon" if region_type == 0 else "Intron" if region_type == 2 else "Intergenic")
        if region_type == 8: # Intergenic
            p.line(
                x=[start, end],
                y=[prediction_track_pos, prediction_track_pos],
                line_width=1,
                color=anno_colors[region_type],
                alpha=0.6,
                legend_label=legend_label,
            )
        elif region_type in anno_colors: # Exon or Intron
            p.quad(
                top=[prediction_track_pos + track_height],
                bottom=[prediction_track_pos - track_height],
                left=[start],
                right=[end],
                color=anno_colors[region_type],
                alpha=0.6,
                legend_label=legend_label,
            )

    # --- Plot Reading Frame Track (Optional) ---
    if grouped_reading_frame is not None:
        for region_type, start, end in grouped_reading_frame:
            # Only plot if the region_type corresponds to a defined frame (0, 1, 2)
            if region_type in rf_colors:
                p.quad(
                    top=[reading_frame_track_pos + track_height],
                    bottom=[reading_frame_track_pos - track_height],
                    left=[start],
                    right=[end],
                    color=rf_colors[region_type],
                    alpha=0.7, # Slightly different alpha maybe
                    legend_label=f"Frame {int(region_type)}"
                )
            # else: # Optionally handle np.inf, e.g., draw a thin grey line
            #     p.line(x=[start, end], y=[reading_frame_track_pos, reading_frame_track_pos], line_width=1, color="lightgrey", alpha=0.5)
            # Currently, np.inf regions are simply not plotted on this track

    # --- Configure Axes ---
    ticker_positions = [prediction_track_pos, ground_truth_track_pos]
    label_overrides = {
        prediction_track_pos: "Prediction",
        ground_truth_track_pos: "Ground Truth",
    }

    if reading_frame is not None:
        ticker_positions.append(reading_frame_track_pos)
        label_overrides[reading_frame_track_pos] = "Reading Frame"

    # Sort positions for a cleaner axis
    ticker_positions.sort(reverse=True) # Show GT on top

    # Calculate dynamic y-range based on track positions and height
    min_y = min(ticker_positions) - track_height - 0.1 # Add buffer below lowest track
    max_y = max(ticker_positions) + track_height + 0.1 # Add buffer above highest track
    p.y_range = Range1d(min_y, max_y)

    p.yaxis.ticker = ticker_positions
    p.yaxis.major_label_overrides = label_overrides
    p.yaxis.major_tick_line_color = None # Hide y-axis ticks
    p.yaxis.minor_tick_line_color = None
    p.ygrid.grid_line_color = None # Hide horizontal grid lines

    # --- Configure Legend ---
    # Bokeh automatically handles unique legend entries
    p.legend.location = "top_left"
    p.legend.title = "Annotations & Frames"
    p.legend.label_text_font_size = "8pt"
    p.legend.click_policy = "hide"
    p.legend.glyph_height = 10 # Make legend glyphs smaller
    p.legend.glyph_width = 10
    p.legend.label_height = 10

    # --- Output ---
    output_file("genome_annotation_comparison.html")
    show(p)

# --- Example Usage ---

# Sample Data (replace with your actual data)
seq_length = 200
# Ground Truth: Exon (0), Intron (2), Intergenic (8)
gt = np.array(
    [8]*20 + [0]*30 + [2]*15 + [0]*25 + [8]*10 + [0]*40 + [2]*20 + [0]*10 + [8]*30
)
# Prediction: Similar structure, maybe with slight differences
pred = np.array(
    [8]*22 + [0]*28 + [2]*16 + [0]*24 + [8]*12 + [0]*38 + [2]*21 + [0]*9 + [8]*30
)
# Reading Frame: 0, 1, 2 in exons, np.inf elsewhere
rf = np.full(seq_length, np.inf)
rf[20:50] = np.tile([0, 1, 2], 10) # Frame 0, 1, 2 repeating in first exon
rf[65:90] = np.tile([1, 2, 0], int(np.ceil(25/3)))[:25] # Frame 1, 2, 0 repeating in second exon
rf[100:140] = np.tile([2, 0, 1], int(np.ceil(40/3)))[:40] # Frame 2, 0, 1 repeating in third exon
rf[160:170] = np.tile([0, 1, 2], int(np.ceil(10/3)))[:10] # Frame 0, 1, 2 repeating in fourth exon

# --- Generate Plots ---

# Plot with all three tracks
print("Generating plot with Ground Truth, Prediction, and Reading Frame tracks...")
plot_pred_vs_gt(gt, pred, reading_frame=rf)

# Plot with only ground truth and prediction
# print("\nGenerating plot with only Ground Truth and Prediction tracks...")
# plot_pred_vs_gt(gt, pred) # Call without the reading_frame argument