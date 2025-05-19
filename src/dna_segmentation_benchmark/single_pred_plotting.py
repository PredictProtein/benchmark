import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Range1d, Text # Import Text for alternative Y-axis labels
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


TYPE_COLORS = {0: "lightgreen", 2: "lightblue", 8: "black"}
RF_COLORS = {
    0: "#66c2a5",  # Medium Teal/Green (Signifies 'Correct' Reading Frame)
    1: "#fc8d62",  # Medium Orange (Signifies 'Incorrect Frame 1')
    2: "#8da0cb"  # Medium Blue/Purple (Signifies 'Incorrect Frame 2')
}
MISMATCH_HIGHLIGHT_COLOR = "red"

# Helper function to group consecutive identical elements (remains the same)
def group_annotation(annotation):
    """Efficiently groups consecutive identical elements in a numpy array."""
    # Ensure handling of empty arrays or arrays with a single value
    if len(annotation) == 0:
        return []
    if len(annotation) == 1:
         return [(annotation[0], 0, 1)]

    # Find indices where the value changes compared to the previous element
    # np.where returns a tuple; get the array of indices [0]
    # Adding 1 to the indices from diff on sliced array to get correct original indices
    change_points = np.where(annotation[1:] != annotation[:-1])[0] + 1

    # The start indices of the groups are 0, followed by the change points
    group_start_indices = np.concatenate(([0], change_points))

    # The end index of a group is the start index of the next group.
    # The last group ends at the total length of the annotation array.
    group_end_indices = np.append(change_points, len(annotation))

    return [
        (annotation[start], start, end)
        for start, end in zip(group_start_indices, group_end_indices)
    ]

def plot_pred_vs_gt_enhanced(
    ground_truth: np.array,
    prediction: np.array,
    labels,
    reading_frame: np.array = None, # Optional reading frame array
    nuc_label_colors : dict = TYPE_COLORS

):
    """
    Plots genome annotation tracks for ground truth, prediction, and optionally
    reading frames, highlighting discrepancies between ground truth and prediction.

    Args:
        ground_truth (np.array): Array representing ground truth annotations.
                                 Expected values: 0 (exon), 2 (intron), 8 (intergenic).
        prediction (np.array): Array representing predicted annotations.
                               Expected values: 0 (exon), 2 (intron), 8 (intergenic).
        reading_frame (np.array, optional): Array representing reading frames.
                                             Expected values: 0, 1, 2 for frames,
                                             np.inf or other values for non-coding/no match.
                                             Defaults to None.

    Highlighting Logic:
        - Prediction segments (quads or lines) are outlined/colored red and thickened
          if the predicted type differs from the ground truth type at *any* point
          within that segment's range.
    Hover Tool:
        - Shows position, segment range [start, end), and type.
        - For prediction segments, indicates "(Match)" or "(Mismatch)".
    """
    # --- Input Validation and Setup ---
    # Added conversion to numpy array if not already, for consistency
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)

    assert len(labels) == len(nuc_label_colors), "Number of labels must match number of colors."

    if ground_truth.shape != prediction.shape:
        raise ValueError("Ground truth and prediction arrays must have the same shape.")

    array_length = len(ground_truth)

    if reading_frame is not None:
        reading_frame = np.asarray(reading_frame)
        if reading_frame.shape != ground_truth.shape:
            raise ValueError("Reading frame array must have the same shape as ground truth.")

    # --- Define Colors and Styles ---
    # Using slightly more descriptive variable names

    # Line widths
    LINE_WIDTH_QUAD_MATCH = 0.5 # Thin outline for matched quads
    LINE_WIDTH_LINE_MATCH = 1.0
    LINE_WIDTH_MISMATCH = 2.0

    BASE_ALPHA = 0.7
    RF_ALPHA = BASE_ALPHA + 0.1 # Slightly less transparent maybe

    # --- Group Annotations ---
    # These will now contain tuples (value, start_index, end_index)
    grouped_ground_truth = group_annotation(ground_truth)
    grouped_prediction = group_annotation(prediction)
    grouped_reading_frame = None
    if reading_frame is not None:
        # Filter out non-plotting values (like np.inf) from reading frame grouping
        grouped_reading_frame = [
            (val, start, end) for val, start, end in group_annotation(reading_frame)
            if val in RF_COLORS # Only keep groups with defined colors
        ]


    # --- Calculate Mismatches (Point-wise) ---
    # This array indicates True at positions where GT != Pred
    is_mismatch_array = (ground_truth != prediction)

    # --- Prepare Data Sources for Bokeh ---
    # Initialize dictionaries that will hold list data
    # Renamed for clarity
    gt_quad_data = {'left': [], 'right': [], 'top': [], 'bottom': [], 'color': [], 'alpha': [], 'type_str': [], 'start': [], 'end': []}
    gt_line_data = {'x': [], 'y': [], 'color': [], 'alpha': [], 'line_width': [], 'type_str': [], 'start': [], 'end': []}
    pred_quad_data = {'left': [], 'right': [], 'top': [], 'bottom': [], 'color': [], 'alpha': [], 'line_color': [], 'line_width': [], 'type_str': [], 'start': [], 'end': [], 'status': []}
    pred_line_data = {'x': [], 'y': [], 'color': [], 'alpha': [], 'line_width': [], 'type_str': [], 'start': [], 'end': [], 'status': []}
    rf_quad_data = {'left': [], 'right': [], 'top': [], 'bottom': [], 'color': [], 'alpha': [], 'type_str': [], 'start': [], 'end': []}

    # --- Define Track Positions and Height ---
    # Using more descriptive names
    GT_TRACK_Y = 1.0
    PRED_TRACK_Y = 0.66
    RF_TRACK_Y = 0.33 # Placed higher for separation
    TRACK_HEIGHT = 0.30 # Increased height slightly for better visibility

    # Y-coordinates for the top and bottom of quads based on track position and height
    get_quad_top = lambda y_pos: y_pos + TRACK_HEIGHT / 2
    get_quad_bottom = lambda y_pos: y_pos - TRACK_HEIGHT / 2


    # --- Populate Ground Truth Data ---
    for region_type, start, end in grouped_ground_truth:
        type_name = "Exon" if region_type == 0 else "Intron" if region_type == 2 else "Intergenic"
        type_str = f"GT: {type_name}"
        color = nuc_label_colors.get(region_type, "gray")

        if region_type == 8: # Intergenic line
            ds = gt_line_data
            ds['x'].append([start, end])
            ds['y'].append([GT_TRACK_Y, GT_TRACK_Y]) # Use track position for Y
            ds['color'].append(color)
            ds['alpha'].append(BASE_ALPHA)
            ds['line_width'].append(LINE_WIDTH_LINE_MATCH)
            ds['type_str'].append(type_str)
            ds['start'].append(start)
            ds['end'].append(end)
        elif region_type in nuc_label_colors: # Exon/Intron quad (0 or 2)
            ds = gt_quad_data
            ds['left'].append(start)
            ds['right'].append(end)
            ds['top'].append(get_quad_top(GT_TRACK_Y)) # Calculate top/bottom
            ds['bottom'].append(get_quad_bottom(GT_TRACK_Y))
            ds['color'].append(color)
            ds['alpha'].append(BASE_ALPHA)
            ds['type_str'].append(type_str)
            ds['start'].append(start)
            ds['end'].append(end)

    # --- Populate Prediction Data (with Mismatch Highlighting) ---
    for region_type, start, end in grouped_prediction:
        type_name = "Exon" if region_type == 0 else "Intron" if region_type == 2 else "Intergenic"
        type_str_base = f"Pred: {type_name}"

        # Check if any point within this predicted segment mismatches GT
        # Handle case where start >= end (shouldn't happen with group_annotation, but good practice)
        segment_mismatches = np.any(is_mismatch_array[start:end]) if start < end else False
        status_str = " (Mismatch)" if segment_mismatches else " (Match)"
        type_str = type_str_base + status_str

        color = nuc_label_colors.get(region_type, "gray") # Base color for fill

        if region_type == 8: # Intergenic line
            ds = pred_line_data
            ds['x'].append([start, end])
            ds['y'].append([PRED_TRACK_Y, PRED_TRACK_Y]) # Use track position for Y
            ds['color'].append(MISMATCH_HIGHLIGHT_COLOR if segment_mismatches else color) # Line color based on mismatch
            ds['alpha'].append(BASE_ALPHA)
            ds['line_width'].append(LINE_WIDTH_MISMATCH if segment_mismatches else LINE_WIDTH_LINE_MATCH)
            ds['type_str'].append(type_str)
            ds['start'].append(start)
            ds['end'].append(end)
            ds['status'].append(status_str)
        elif region_type in nuc_label_colors: # Exon/Intron quad (0 or 2)
            ds = pred_quad_data
            ds['left'].append(start)
            ds['right'].append(end)
            ds['top'].append(get_quad_top(PRED_TRACK_Y)) # Calculate top/bottom
            ds['bottom'].append(get_quad_bottom(PRED_TRACK_Y))
            ds['color'].append(color) # Fill color (usually doesn't change based on mismatch)
            ds['alpha'].append(BASE_ALPHA)
            ds['line_color'].append(MISMATCH_HIGHLIGHT_COLOR if segment_mismatches else 'black') # Outline color
            ds['line_width'].append(LINE_WIDTH_MISMATCH if segment_mismatches else LINE_WIDTH_QUAD_MATCH) # Line width
            ds['type_str'].append(type_str)
            ds['start'].append(start)
            ds['end'].append(end)
            ds['status'].append(status_str)

    # --- Populate Reading Frame Data (Optional) ---
    if grouped_reading_frame is not None:
        for region_type, start, end in grouped_reading_frame:
             # Note: filtering for region_type in RF_COLORS is done during grouping now
            ds = rf_quad_data
            ds['left'].append(start)
            ds['right'].append(end)
            ds['top'].append(get_quad_top(RF_TRACK_Y)) # Calculate top/bottom
            ds['bottom'].append(get_quad_bottom(RF_TRACK_Y))
            ds['color'].append(RF_COLORS[region_type])
            ds['alpha'].append(RF_ALPHA)
            ds['type_str'].append(f"RF: Frame {int(region_type)}")
            ds['start'].append(start)
            ds['end'].append(end)


    # --- Convert dictionaries to ColumnDataSources ---
    gt_quad_source = ColumnDataSource(gt_quad_data)
    gt_line_source = ColumnDataSource(gt_line_data)
    pred_quad_source = ColumnDataSource(pred_quad_data)
    pred_line_source = ColumnDataSource(pred_line_data)
    rf_quad_source = ColumnDataSource(rf_quad_data)

    # --- Create Bokeh Figure ---
    plot_title = "Genome Annotation Comparison (Red outline/line = Prediction Mismatch)"
    p = figure(
        height=800,
        width=1700,
        title=plot_title,
        tools="pan,wheel_zoom,box_zoom,reset,save", # Removed 'hover' here, add it below
        x_axis_label="Position",
        # y_axis_label="Tracks", # Removed generic y-axis label
        active_scroll="wheel_zoom"
    )

    # --- Plot Glyphs from Sources and Capture Renderers ---
    # Ground Truth
    gt_quads_renderer = p.quad(source=gt_quad_source, name="gt_quads", legend_label="GT Exon/Intron",
                              left='left', right='right', top='top', bottom='bottom', color='color', alpha='alpha',
                              line_color="black", line_width=LINE_WIDTH_QUAD_MATCH)

    gt_lines_renderer = p.multi_line(source=gt_line_source, name="gt_lines", legend_label="GT Intergenic",
                                     xs='x', ys='y', color='color', alpha='alpha', line_width='line_width')

    # Prediction (with potential highlighting)
    pred_quads_renderer = p.quad(source=pred_quad_source, name="pred_quads", legend_label="Pred Exon/Intron",
                                 left='left', right='right', top='top', bottom='bottom', color='color', alpha='alpha',
                                 line_color='line_color', line_width='line_width') # line_color/width from source

    pred_lines_renderer = p.multi_line(source=pred_line_source, name="pred_lines", legend_label="Pred Intergenic",
                                       xs='x', ys='y', color='color', alpha='alpha', line_width='line_width')

    # Reading Frame (Optional)
    rf_quads_renderer = None
    if grouped_reading_frame: # Check if there's data to plot after filtering
        rf_quads_renderer = p.quad(source=rf_quad_source, name="rf_quads", legend_label="Reading Frame",
                                   left='left', right='right', top='top', bottom='bottom', color='color', alpha='alpha',
                                   line_color="black", line_width=LINE_WIDTH_QUAD_MATCH)

    # --- Configure Axes (Using Fixed Ticks and Labels) ---
    # Define positions for the custom Y-axis labels
    track_y_positions = [GT_TRACK_Y, PRED_TRACK_Y]
    track_labels = ["Ground Truth", "Prediction"]

    if rf_quads_renderer is not None: # Only add RF if it's plotted
        track_y_positions.append(RF_TRACK_Y)
        track_labels.append("Reading Frame")

    # Sort by Y position if needed, or just ensure order matches labels
    # Sorting ensures ticks appear in ascending/descending order if not already
    sorted_indices = np.argsort(track_y_positions)
    sorted_y_positions = np.array(track_y_positions)[sorted_indices]
    sorted_labels = np.array(track_labels)[sorted_indices]

    # Create a dictionary for the overrides {position: label}
    label_overrides = dict(zip(sorted_y_positions, sorted_labels))


    p.yaxis.ticker = sorted_y_positions # Set ticks at the track positions
    p.yaxis.major_label_overrides = label_overrides # Use labels for these ticks
    p.yaxis.major_tick_line_color = None # Hide tick lines
    p.yaxis.minor_tick_line_color = None
    p.ygrid.grid_line_color = 'lightgray' # Add horizontal grid lines at track positions
    p.ygrid.grid_line_alpha = 0.5


    # Set Y-range to comfortably fit tracks and labels
    min_y = min(track_y_positions) - TRACK_HEIGHT/2 - 0.5 # Add some padding
    max_y = max(track_y_positions) + TRACK_HEIGHT/2 + 0.5 # Add some padding
    p.y_range = Range1d(min_y, max_y)

    # --- Configure Hover Tool ---
    # Collect the renderers you want the HoverTool to target
    # Alternative list building using if statement for clarity
    targeted_renderers = [
        gt_quads_renderer,
        gt_lines_renderer,
        pred_quads_renderer,
        pred_lines_renderer,
    ]
    if rf_quads_renderer is not None:
        targeted_renderers.append(rf_quads_renderer)


    hover = HoverTool(
        renderers=targeted_renderers, # Pass the list of renderer objects here
        tooltips="""
            <div>
                <span style="font-weight: bold;">Position:</span> @$x{0,0} <br>
                <span style="font-weight: bold;">Region:</span> [@start{0,0}, @end{0,0}) <br>
                <span style="font-weight: bold;">Type:</span> @type_str
            </div>
        """,
        mode='mouse'
    )
    p.add_tools(hover) # Add hover tool to the plot

    # --- Configure Legend ---
    p.legend.location = "top_left"
    p.legend.title = "Annotations"
    p.legend.label_text_font_size = "9pt" # Slightly larger text
    p.legend.click_policy = "hide" # Allows hiding glyphs by clicking legend
    p.legend.glyph_height = 12 # Slightly larger glyphs in legend
    p.legend.glyph_width = 12
    p.legend.spacing = 5 # Space between legend items
    p.legend.margin = 10 # Margin around legend


    # --- Output ---
    output_file("genome_annotation_comparison_enhanced.html", title="Genome Annotation Comparison")
    show(p)


def plot_error_summary_bar(error_dict: dict, title: str = "Total Prediction Errors"):
    """
    Plots the total number of errors for each error type as a horizontal bar plot.

    Args:
        error_dict: A dictionary where keys are error type names (str)
                    and values are lists of error instances.
        title: The title for the plot.
    """
    total_error_dict = {key: len(value) for key, value in error_dict.items()}
    if not total_error_dict:
        print("No error data to plot in plot_error_summary_bar.")
        return

    total_error_df = pd.DataFrame(total_error_dict.items(), columns=["Error Type", "Count"])
    total_error_df = total_error_df.sort_values(by="Count", ascending=True)

    plt.figure(figsize=(16, 10))
    ax = sns.barplot(data=total_error_df, y="Error Type", x="Count", color="skyblue")

    plt.title(title, fontsize=16)
    plt.xlabel("Number of Errors", fontsize=12)
    plt.ylabel("Error Type", fontsize=12)

    # Annotate each bar with its total value
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_width())}",
            (p.get_width(), p.get_y() + p.get_height() / 2),
            ha="left",
            va="center",
            fontsize=10,
            color="black",
            xytext=(5, 0),  # Offset text from the bar
            textcoords="offset points",
        )

    plt.show()