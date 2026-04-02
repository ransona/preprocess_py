import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import organise_paths
import pickle
from datetime import datetime
import harp


def _as_1d_float(array):
    return np.atleast_1d(np.squeeze(np.asarray(array, dtype=float)))


def _guess_missing_flip_times(reference_flips, target_flips):
    reference_flips = _as_1d_float(reference_flips)
    target_flips = _as_1d_float(target_flips)

    missing_count = len(reference_flips) - len(target_flips)
    if missing_count <= 0 or len(reference_flips) < 3 or len(target_flips) < 2:
        return np.array([])

    reference_rel = reference_flips - reference_flips[0]
    target_rel = target_flips - target_flips[0]
    kept_indices = np.arange(len(reference_flips))
    guessed_times = []

    for _ in range(missing_count):
        best_candidate = None
        for remove_pos in range(len(kept_indices)):
            candidate_indices = np.delete(kept_indices, remove_pos)
            if len(candidate_indices) != len(target_flips):
                continue

            candidate_ref_rel = reference_rel[candidate_indices]
            if np.allclose(candidate_ref_rel, candidate_ref_rel[0]):
                continue

            slope, intercept = np.polyfit(candidate_ref_rel, target_rel, 1)
            fit_error = np.median(np.abs((slope * candidate_ref_rel + intercept) - target_rel))

            if best_candidate is None or fit_error < best_candidate['fit_error']:
                best_candidate = {
                    'remove_pos': remove_pos,
                    'fit_error': fit_error,
                    'slope': slope,
                    'intercept': intercept,
                }

        if best_candidate is None:
            break

        missing_index = kept_indices[best_candidate['remove_pos']]
        guessed_rel_time = (best_candidate['slope'] * reference_rel[missing_index]) + best_candidate['intercept']
        guessed_times.append(target_flips[0] + guessed_rel_time)
        kept_indices = np.delete(kept_indices, best_candidate['remove_pos'])

    return np.sort(np.asarray(guessed_times))


def _guess_missing_flips_by_system(flip_dict, preferred_reference='bv'):
    counts = {name: len(_as_1d_float(flips)) for name, flips in flip_dict.items()}
    max_count = max(counts.values())
    if max_count == 0:
        return {name: np.array([]) for name in flip_dict}, None

    if counts.get(preferred_reference, 0) == max_count:
        reference_name = preferred_reference
    else:
        reference_name = sorted(
            [name for name, count in counts.items() if count == max_count],
            key=lambda name: (name != preferred_reference, name),
        )[0]

    reference_flips = _as_1d_float(flip_dict[reference_name])
    guessed = {}
    for name, flips in flip_dict.items():
        current_flips = _as_1d_float(flips)
        if len(current_flips) < len(reference_flips):
            guessed[name] = _guess_missing_flip_times(reference_flips, current_flips)
        else:
            guessed[name] = np.array([])

    return guessed, reference_name


def _build_interp_pair(reference_flips, target_flips, debug=False, pair_name='flip pair'):
    reference_flips = _as_1d_float(reference_flips)
    target_flips = _as_1d_float(target_flips)

    if len(reference_flips) == len(target_flips):
        return reference_flips, target_flips

    if debug:
        if len(target_flips) < len(reference_flips):
            guessed_target = _guess_missing_flip_times(reference_flips, target_flips)
            if len(guessed_target) > 0:
                target_flips = np.sort(np.concatenate([target_flips, guessed_target]))
        elif len(reference_flips) < len(target_flips):
            guessed_reference = _guess_missing_flip_times(target_flips, reference_flips)
            if len(guessed_reference) > 0:
                reference_flips = np.sort(np.concatenate([reference_flips, guessed_reference]))

        if len(reference_flips) != len(target_flips):
            min_len = min(len(reference_flips), len(target_flips))
            print(f'Debug mode enabled: trimming {pair_name} to {min_len} matched flips for interpolation.')
            reference_flips = reference_flips[:min_len]
            target_flips = target_flips[:min_len]

    if len(reference_flips) != len(target_flips):
        raise ValueError(f'{pair_name} arrays must be equal in length for interpolation')

    return reference_flips, target_flips


def _find_excluded_flips(raw_flips, filtered_flips, atol=1e-9):
    raw_flips = _as_1d_float(raw_flips)
    filtered_flips = _as_1d_float(filtered_flips)
    if len(raw_flips) == 0:
        return np.array([])
    if len(filtered_flips) == 0:
        return raw_flips.copy()

    kept_mask = np.zeros(len(raw_flips), dtype=bool)
    for idx, raw_flip in enumerate(raw_flips):
        kept_mask[idx] = np.any(np.isclose(raw_flip, filtered_flips, atol=atol, rtol=0))
    return raw_flips[~kept_mask]


def _build_excluded_details(
    raw_flips,
    kept_indices,
    min_pulse_width,
    max_pulse_width,
    interval_source_flips=None,
    reason_source_name=None,
    timeline_reference_flips=None,
):
    raw_flips = _as_1d_float(raw_flips)
    kept_indices = np.asarray(kept_indices, dtype=int)
    interval_source_flips = raw_flips if interval_source_flips is None else _as_1d_float(interval_source_flips)
    reason_source_name = 'same system' if reason_source_name is None else reason_source_name
    timeline_reference_flips = None if timeline_reference_flips is None else _as_1d_float(timeline_reference_flips)

    kept_mask = np.zeros(len(raw_flips), dtype=bool)
    valid_kept_indices = kept_indices[(kept_indices >= 0) & (kept_indices < len(raw_flips))]
    kept_mask[valid_kept_indices] = True

    details = []
    interval_diffs = np.diff(interval_source_flips)
    for idx, flip_time in enumerate(raw_flips):
        if kept_mask[idx]:
            continue

        if idx >= len(interval_diffs):
            interval = np.nan
            reason = f'excluded because there is no following interval available in {reason_source_name}'
        else:
            interval = interval_diffs[idx]
            if interval <= min_pulse_width:
                reason = (
                    f'excluded because next interval in {reason_source_name} is too short '
                    f'({interval:.4f}s <= {min_pulse_width:.4f}s)'
                )
            elif np.isfinite(max_pulse_width) and interval >= max_pulse_width:
                reason = (
                    f'excluded because next interval in {reason_source_name} is too long '
                    f'({interval:.4f}s >= {max_pulse_width:.4f}s)'
                )
            else:
                reason = f'excluded by filter despite interval in {reason_source_name} being within bounds'

        if timeline_reference_flips is None:
            timeline_time = float(flip_time)
        elif idx < len(timeline_reference_flips):
            timeline_time = float(timeline_reference_flips[idx])
        else:
            timeline_time = np.nan

        details.append({
            'time': float(flip_time),
            'raw_time': float(flip_time),
            'timeline_time': timeline_time,
            'interval': float(interval) if np.isfinite(interval) else np.nan,
            'reason': reason,
            'flip_index': idx,
            'system_name': reason_source_name,
        })

    return details


def _append_last_index(indices, length):
    indices = np.asarray(indices, dtype=int)
    if length == 0:
        return indices
    last_idx = length - 1
    if last_idx not in indices:
        indices = np.append(indices, last_idx)
    return np.sort(indices)


def _print_excluded_details(excluded_flip_details):
    print('Debug excluded pulse list:')
    any_excluded = False
    for system_name in ['tl_pd', 'tl_bv', 'bv', 'harp']:
        details = excluded_flip_details.get(system_name, [])
        print(f'  {system_name}: {len(details)} excluded')
        for detail in details:
            any_excluded = True
            interval_text = 'NA' if np.isnan(detail['interval']) else f'{detail["interval"]:.6f}s'
            timeline_text = 'NA' if np.isnan(detail['timeline_time']) else f'{detail["timeline_time"]:.6f}s'
            print(
                '    '
                f'idx={detail["flip_index"]} '
                f'interval={interval_text} '
                f'timeline_time={timeline_text} '
                f'reason={detail["reason"]}'
            )
    if not any_excluded:
        print('  none')


def _make_debug_panel(time, signal, flips, guessed_flips, title, xlabel, system_name=None, filtered_flips=None, excluded_flips=None, excluded_details=None):
    return {
        'time': _as_1d_float(time),
        'signal': _as_1d_float(signal),
        'flips': _as_1d_float(flips),
        'guessed_flips': _as_1d_float(guessed_flips) if len(np.atleast_1d(guessed_flips)) > 0 else np.array([]),
        'filtered_flips': _as_1d_float(filtered_flips) if filtered_flips is not None and len(np.atleast_1d(filtered_flips)) > 0 else np.array([]),
        'excluded_flips': _as_1d_float(excluded_flips) if excluded_flips is not None and len(np.atleast_1d(excluded_flips)) > 0 else np.array([]),
        'excluded_details': [] if excluded_details is None else excluded_details,
        'title': title,
        'xlabel': xlabel,
        'system_name': system_name if system_name is not None else title,
    }


def _prepare_debug_panel(panel, debug_params, apply_time_range=True):
    time_plot = panel['time'].copy()
    flips_plot = panel['flips'].copy()
    guessed_plot = panel['guessed_flips'].copy()
    filtered_plot = panel['filtered_flips'].copy()
    excluded_plot = panel['excluded_flips'].copy()
    excluded_details = [dict(detail) for detail in panel['excluded_details']]

    if debug_params['align_to_first_pulse'] and len(flips_plot) > 0:
        alignment_offset = flips_plot[0]
        time_plot = time_plot - alignment_offset
        flips_plot = flips_plot - alignment_offset
        guessed_plot = guessed_plot - alignment_offset
        filtered_plot = filtered_plot - alignment_offset
        excluded_plot = excluded_plot - alignment_offset
        for detail in excluded_details:
            detail['time'] = detail['time'] - alignment_offset

    debug_time_range = debug_params['time_range']
    if (debug_time_range is None) or (not apply_time_range):
        time_mask = np.ones_like(time_plot, dtype=bool)
    else:
        debug_start, debug_end = debug_time_range
        if debug_end <= debug_start:
            raise ValueError('debug_params["time_range"] must have end > start')
        if debug_params['align_to_first_pulse'] and len(flips_plot) > 0:
            debug_start = min(debug_start, np.min(time_plot))
        time_mask = (time_plot >= debug_start) & (time_plot <= debug_end)
        if not time_mask.any():
            raise ValueError('debug_params["time_range"] does not overlap the available recording times')
        flips_plot = flips_plot[(flips_plot >= debug_start) & (flips_plot <= debug_end)]
        guessed_plot = guessed_plot[(guessed_plot >= debug_start) & (guessed_plot <= debug_end)]
        filtered_plot = filtered_plot[(filtered_plot >= debug_start) & (filtered_plot <= debug_end)]
        excluded_plot = excluded_plot[(excluded_plot >= debug_start) & (excluded_plot <= debug_end)]
        excluded_details = [
            detail for detail in excluded_details
            if debug_start <= detail['time'] <= debug_end
        ]

    return {
        **panel,
        'time_plot': time_plot,
        'time_mask': time_mask,
        'flips_plot': flips_plot,
        'guessed_plot': guessed_plot,
        'filtered_plot': filtered_plot,
        'excluded_plot': excluded_plot,
        'excluded_details_plot': excluded_details,
    }


def _draw_debug_panels(axes, panels, line_colors, flip_colors):
    for ax, panel, line_color, flip_color in zip(axes, panels, line_colors, flip_colors):
        signal_window = panel['signal'][panel['time_mask']]
        y_min = np.min(signal_window)
        y_max = np.max(signal_window)
        if np.isclose(y_min, y_max):
            y_pad = 1 if y_min == 0 else abs(y_min) * 0.05
            y_min -= y_pad
            y_max += y_pad

        ax.plot(
            panel['time_plot'][panel['time_mask']],
            signal_window,
            color=line_color,
            linewidth=0.8,
            label=panel['title'],
        )
        if len(panel['flips_plot']) > 0:
            ax.vlines(
                panel['flips_plot'],
                ymin=y_min,
                ymax=y_max,
                color=flip_color,
                alpha=0.25,
                linewidth=0.8,
                label='Detected flips',
            )
        if len(panel['guessed_plot']) > 0:
            ax.vlines(
                panel['guessed_plot'],
                ymin=y_min,
                ymax=y_max,
                color='magenta',
                alpha=0.9,
                linewidth=1.2,
                linestyles='--',
                label='Guessed missing flips',
            )
        if len(panel['filtered_plot']) > 0:
            ax.vlines(
                panel['filtered_plot'],
                ymin=y_min,
                ymax=y_max,
                color='tab:cyan',
                alpha=0.35,
                linewidth=0.8,
                linestyles='-.',
                label='Kept after filtering',
            )
        if len(panel['excluded_plot']) > 0:
            ax.vlines(
                panel['excluded_plot'],
                ymin=y_min,
                ymax=y_max,
                color='tab:orange',
                alpha=0.95,
                linewidth=1.4,
                label='Excluded flips',
            )
        ax.set_title(panel['title'])
        ax.set_xlabel(panel['xlabel'])
        ax.set_ylabel('Signal')
        ax.legend(loc='upper right')


def _connect_axis_sync(fig, axis_groups):
    sync_state = {'updating': False}

    def sync_xlimits(changed_ax):
        if sync_state['updating']:
            return
        sync_state['updating'] = True
        try:
            group = None
            for group_axes in axis_groups.values():
                if changed_ax in group_axes:
                    group = group_axes
                    break
            if group is None:
                return
            new_xlim = changed_ax.get_xlim()
            for ax in group:
                if ax is not changed_ax:
                    ax.set_xlim(new_xlim)
            fig.canvas.draw_idle()
        finally:
            sync_state['updating'] = False

    for group_axes in axis_groups.values():
        for ax in group_axes:
            ax.callbacks.connect('xlim_changed', sync_xlimits)


def _build_excluded_events(excluded_panels):
    events = []
    for panel_idx, panel in enumerate(excluded_panels):
        for detail in panel['excluded_details_plot']:
            events.append({
                'panel_idx': panel_idx,
                'system_name': panel['system_name'],
                'time': detail['time'],
                'reason': detail['reason'],
            })
    events.sort(key=lambda event: event['time'])
    return events


def _compute_pairwise_residuals(reference_flips, target_flips):
    reference_flips = _as_1d_float(reference_flips)
    target_flips = _as_1d_float(target_flips)
    n = min(len(reference_flips), len(target_flips))
    if n < 2:
        return np.array([]), np.array([])
    ref_rel = reference_flips[:n] - reference_flips[0]
    tgt_rel = target_flips[:n] - target_flips[0]
    residuals = tgt_rel - ref_rel
    event_idx = np.arange(n)
    return event_idx, residuals


def _format_count_summary(raw_flip_times, filtered_flip_times):
    systems = ['bv', 'tl_bv', 'tl_pd', 'harp']
    lines = ['Counts: raw -> filtered']
    for system in systems:
        raw_count = len(raw_flip_times[system])
        filtered_count = len(filtered_flip_times[system])
        lines.append(f'{system}: {raw_count} -> {filtered_count} (excluded {raw_count - filtered_count})')
    return '\n'.join(lines)


def _show_pulse_diagnostics(raw_flip_times, filtered_flip_times, min_pulse_width, max_pulse_width):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.subplots_adjust(top=0.9, hspace=0.45, wspace=0.3)
    fig.suptitle('Pulse Diagnostics')

    axes[0, 0].axis('off')
    axes[0, 0].text(
        0,
        1,
        _format_count_summary(raw_flip_times, filtered_flip_times),
        ha='left',
        va='top',
        fontsize=10,
        family='monospace',
    )

    interval_specs = [
        (axes[0, 1], ('bv', 'tl_bv'), 'Intervals: BV CSV vs Timeline Bonvision'),
        (axes[1, 0], ('tl_pd', 'harp'), 'Intervals: Timeline PD vs Harp PD'),
    ]
    for ax, (sys_a, sys_b), title in interval_specs:
        intervals_a = np.diff(_as_1d_float(raw_flip_times[sys_a]))
        intervals_b = np.diff(_as_1d_float(raw_flip_times[sys_b]))
        if len(intervals_a) > 0:
            ax.hist(intervals_a, bins=80, alpha=0.5, label=sys_a)
        if len(intervals_b) > 0:
            ax.hist(intervals_b, bins=80, alpha=0.5, label=sys_b)
        ax.axvline(min_pulse_width, color='tab:red', linestyle='--', linewidth=1)
        if np.isfinite(max_pulse_width):
            ax.axvline(max_pulse_width, color='tab:red', linestyle='--', linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('Inter-flip interval (s)')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right')

    residual_pairs = [
        ('bv', 'tl_bv', 'Residuals: BV CSV -> Timeline Bonvision'),
        ('tl_bv', 'tl_pd', 'Residuals: Timeline Bonvision -> Timeline PD'),
        ('tl_pd', 'harp', 'Residuals: Timeline PD -> Harp PD'),
    ]
    residual_axes = [axes[1, 1], axes[2, 0], axes[2, 1]]
    for ax, (ref_name, tgt_name, title) in zip(residual_axes, residual_pairs):
        event_idx, residuals = _compute_pairwise_residuals(raw_flip_times[ref_name], raw_flip_times[tgt_name])
        if len(event_idx) > 0:
            ax.plot(event_idx, residuals * 1000, '.', markersize=3)
            ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel('Event index')
        ax.set_ylabel('Residual (ms)')

    plt.show()


def _show_debug_tabs(expID, raw_panels, filtered_panels, excluded_panels=None, excluded_window=2.0):
    fig, raw_axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    fig.subplots_adjust(top=0.9, bottom=0.14, hspace=0.55)
    fig.suptitle(f'Preprocess BV2 debug traces: {expID}')

    line_colors = ['black', 'tab:blue', 'tab:green', 'tab:purple']
    flip_colors = ['tab:red', 'tab:orange', 'tab:red', 'tab:red']
    _draw_debug_panels(raw_axes, raw_panels, line_colors, flip_colors)

    filtered_axes = []
    for idx, raw_ax in enumerate(raw_axes):
        filtered_ax = fig.add_axes(raw_ax.get_position(), label=f'filtered_debug_{idx}')
        filtered_ax.set_visible(False)
        filtered_axes.append(filtered_ax)

    _draw_debug_panels(filtered_axes, filtered_panels, line_colors, flip_colors)
    axis_groups = {'raw': list(raw_axes), 'filtered': filtered_axes}

    excluded_axes = []
    excluded_events = []
    excluded_counts = {}
    if excluded_panels is not None:
        for idx, raw_ax in enumerate(raw_axes):
            excluded_ax = fig.add_axes(raw_ax.get_position(), label=f'excluded_debug_{idx}')
            excluded_ax.set_visible(False)
            excluded_axes.append(excluded_ax)
        _draw_debug_panels(excluded_axes, excluded_panels, line_colors, flip_colors)
        axis_groups['excluded'] = excluded_axes
        excluded_events = _build_excluded_events(excluded_panels)
        excluded_counts = {
            panel['system_name']: len(panel['excluded_details_plot'])
            for panel in excluded_panels
        }

    _connect_axis_sync(fig, axis_groups)

    raw_button_ax = fig.add_axes([0.60, 0.935, 0.1, 0.04])
    filtered_button_ax = fig.add_axes([0.71, 0.935, 0.1, 0.04])
    raw_button = Button(raw_button_ax, 'Raw')
    filtered_button = Button(filtered_button_ax, 'Filtered')
    if excluded_panels is not None:
        excluded_button_ax = fig.add_axes([0.82, 0.935, 0.14, 0.04])
        excluded_button = Button(
            excluded_button_ax,
            'Excl '
            f'T:{len(excluded_events)} '
            f'PD:{excluded_counts.get("tl_pd", 0)} '
            f'TLBV:{excluded_counts.get("tl_bv", 0)} '
            f'BV:{excluded_counts.get("bv", 0)} '
            f'H:{excluded_counts.get("harp", 0)}'
        )
        prev_button_ax = fig.add_axes([0.60, 0.02, 0.1, 0.04])
        next_button_ax = fig.add_axes([0.71, 0.02, 0.1, 0.04])
        prev_button = Button(prev_button_ax, 'Prev')
        next_button = Button(next_button_ax, 'Next')
        event_text_ax = fig.add_axes([0.02, 0.01, 0.56, 0.08])
        event_text_ax.axis('off')
        event_text = event_text_ax.text(0, 0.98, '', ha='left', va='top', fontsize=9, wrap=True)
    else:
        prev_button = None
        next_button = None
        event_text = None

    active_group = {'name': 'raw'}
    current_event_idx = {'value': 0}

    def update_excluded_controls(group_name):
        if excluded_panels is None:
            return
        prev_button_ax.set_visible(group_name == 'excluded')
        next_button_ax.set_visible(group_name == 'excluded')
        if event_text is not None and group_name != 'excluded':
            event_text.set_text('')

    def show_group(group_name):
        previous_group = active_group['name']
        if previous_group == group_name:
            update_excluded_controls(group_name)
            return

        previous_xlim = axis_groups[previous_group][0].get_xlim()
        for ax in axis_groups[previous_group]:
            ax.set_visible(False)
        for ax in axis_groups[group_name]:
            ax.set_visible(True)
            ax.set_xlim(previous_xlim)
        active_group['name'] = group_name
        if event_text is not None:
            if group_name == 'excluded' and len(excluded_events) > 0:
                show_excluded_event(current_event_idx['value'])
            else:
                event_text.set_text('')
        update_excluded_controls(group_name)
        fig.canvas.draw_idle()

    def show_excluded_event(event_idx):
        if len(excluded_events) == 0:
            if event_text is not None:
                event_text.set_text('Excluded 0/0')
            return

        event_idx = event_idx % len(excluded_events)
        current_event_idx['value'] = event_idx
        event = excluded_events[event_idx]
        x_start = event['time'] - excluded_window
        x_end = event['time'] + excluded_window
        for ax in excluded_axes:
            ax.set_xlim(x_start, x_end)
        if event_text is not None:
            event_text.set_text(
                f'Excluded {event_idx + 1}/{len(excluded_events)}\n'
                f'System: {event["system_name"]} | Time: {event["time"]:.3f}s\n'
                f'Reason: {event["reason"]}'
            )
        fig.canvas.draw_idle()

    raw_button.on_clicked(lambda event: show_group('raw'))
    filtered_button.on_clicked(lambda event: show_group('filtered'))
    if excluded_panels is not None:
        excluded_button.on_clicked(lambda event: show_group('excluded'))
        prev_button.on_clicked(lambda event: show_excluded_event(current_event_idx['value'] - 1))
        next_button.on_clicked(lambda event: show_excluded_event(current_event_idx['value'] + 1))
        prev_button_ax.set_visible(False)
        next_button_ax.set_visible(False)

    plt.show()

def run_preprocess_bv2(userID, expID, debug=False, debug_params=None):
    print('Starting run_preprocess_bv...')
    # filter_timing_pulses = True allows removal of timing pulses with duration < min_pulse_width
    # this is used to deal with random fast alterations that bonsai still sometimes produces
    # which can be detected in the electrical signal but not in the photodiode necessarily
    # we therefore remove flips of < min_pulse_width duration
    filter_flips = True
    min_pulse_width = 0.05 # seconds
    max_pulse_width = np.inf
    encoder_scaling_factor = 0.0005371094

    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')
    # load the stimulus parameter file produced by matlab by the bGUI
    # this includes stim parameters and stimulus order
    try:
        stim_params = loadmat(os.path.join(exp_dir_raw, expID + '_stim.mat'))
    except:
        raise Exception('Stimulus parameter file not found - this experiment was probably from pre-Dec 2021.')
    
    # load timeline
    Timeline = loadmat(os.path.join(exp_dir_raw, expID + '_Timeline.mat'))
    Timeline = Timeline['timelineSession']
    # get timeline file in a usable format after importing to python
    tl_chNames = Timeline['chNames'][0][0][0][0:]
    tl_daqData = Timeline['daqData'][0,0]
    tl_time    = Timeline['time'][0][0]
    tl_time    = np.squeeze(tl_time)
    # load BV data
    frame_events = pd.read_csv(os.path.join(exp_dir_raw, expID + '_FrameEvents.csv'), names=['Frame', 'Timestamp', 'Sync', 'Trial'],
                            header=None, skiprows=[0], dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Sync':np.float32, 'Trial':np.float32})
    bv_encoder = pd.read_csv(os.path.join(exp_dir_raw, expID + '_Encoder.csv'), names=['Frame', 'Timestamp', 'Unknown', 'Encoder'],
                            header=None, skiprows=[0], dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Sync':np.float32, 'Trial':np.float32})    

    # load Harp raw data
    # check if they exist and which version
    if os.path.exists(os.path.join(exp_dir_raw, expID + '_Behavior_Event44.bin')):
        data_read = harp.io.read(os.path.join(exp_dir_raw, expID + '_Behavior_Event44.bin'))
        data_read_np = np.array(data_read)
        # remove first 400 samples of data as they can contain initiatation random signals
        data_read_np[0:200,0] = data_read_np[200,0]
        # harp encoder log was previously summed
        harp_encoder = data_read_np[:,1]
    elif os.path.exists(os.path.join(exp_dir_raw, expID + '_Behavior_44.bin')):
        data_read = harp.io.read(os.path.join(exp_dir_raw, expID + '_Behavior_44.bin'))
        data_read_np = np.array(data_read)
        # remove first 400 samples of data as they can contain initiatation random signals
        data_read_np[0:200,0] = data_read_np[200,0]        
        # harp encoder log was later dif and so do cumsum of difs
        harp_encoder = np.cumsum(data_read_np[:,1])
    else:
        raise Exception('Harp data file not found')
    
    harp_pd = data_read_np[:,0]
    harp_time = np.arange(0, len(harp_pd)/1000, 1/1000)

    # /////////////// DETECTING TIMING PULSES ///////////////

    # Find Harp times when PD flip
    #  Threshold the Harp PD signal and detect flips
    harp_pd_smoothed = pd.Series(harp_pd).rolling(window=20, min_periods=1).mean().values
    harp_pd_smoothed = harp_pd_smoothed - np.min(harp_pd_smoothed)
    harp_pd_high = np.percentile(harp_pd_smoothed, 99)
    harp_pd_low = np.percentile(harp_pd_smoothed, 1)   
    # calculate 'signal to noise' ratio - this will be low if the signal is just noise
    harp_pd_on_off_ratio = (harp_pd_high - harp_pd_low) / harp_pd_low 
    if harp_pd_on_off_ratio > 10:
        harp_pd_valid = True  
    else:
        harp_pd_valid = False   

    harp_pd_threshold = harp_pd_low + ((harp_pd_high - harp_pd_low)*0.5)
    harp_pd_thresholded = np.where(harp_pd_smoothed < harp_pd_threshold, 0, 1)
    transitions = np.diff(harp_pd_thresholded)
    flip_samples = np.where(transitions == 1)[0]
    flip_times_harp = harp_time[flip_samples]
    
    # Find BV times when digital flips
    Timestamp = frame_events['Timestamp'].values
    Sync = frame_events['Sync'].values
    Trial = frame_events['Trial']
    Frame = frame_events['Frame']
    bv_rises = np.where(np.diff(Sync) == 1)[0]
    flip_times_bv_bv = np.squeeze(Timestamp[bv_rises])
    # the Bonvision sync signal is guaranteed to start high at experiment start
    flip_times_bv_bv = np.insert(flip_times_bv_bv, 0, Timestamp[0])

    # Find TL times when PD flips
    tl_pd_ch = np.where(np.isin(tl_chNames, 'Photodiode'))
    tl_pd = np.squeeze(tl_daqData[:, tl_pd_ch])
    # Needs to be smoothed because the photodiode signal is noisy and monitor blanking can further screw it up
    # There should also be a RC filter in the photodiode signal path to smooth high frequency noise
    tl_pd_smoothed = pd.Series(tl_pd).rolling(window=40, min_periods=1).mean().values
    tl_pd_smoothed = tl_pd_smoothed - np.min(tl_pd_smoothed)
    # Calculate threshold for PD signal
    tl_pd_high = np.percentile(tl_pd_smoothed, 99)
    tl_pd_low = np.percentile(tl_pd_smoothed, 1)
    tl_pd_on_off_ratio = (tl_pd_high - tl_pd_low) / tl_pd_low

    if tl_pd_on_off_ratio > 10:
        tl_pd_valid = True
    else:
        tl_pd_valid = False

    tl_pd_threshold = tl_pd_low + ((tl_pd_high - tl_pd_low)*0.5)
    tl_pd_thresholded = np.squeeze(tl_pd_smoothed > tl_pd_threshold).astype(int)
    # Detect rising edges only
    tl_pd_thresholded_diff = np.diff(tl_pd_thresholded)
    flip_times_pd_tl = np.squeeze(tl_time[np.where(tl_pd_thresholded_diff == 1)])

    # Find TL times when BV Digital flips
    tl_bv_ch = np.where(np.isin(tl_chNames, 'Bonvision'))
    tl_bv = np.squeeze(tl_daqData[:, tl_bv_ch])
    # Needs to be smoothed because the photodiode signal is noisy and monitor blanking can further screw it up
    # There should also be a RC filter in the photodiode signal path to smooth high frequency noise
    tl_bv_smoothed = pd.Series(tl_bv).rolling(window=1, min_periods=1).mean().values
    # Calculate threshold for PD signal
    tl_bv_high = np.percentile(tl_bv_smoothed, 99)
    tl_bv_low = np.percentile(tl_bv_smoothed, 1)
    tl_bv_threshold = tl_bv_low + ((tl_bv_high - tl_bv_low)/2)
    tl_bv_thresholded = np.squeeze(tl_bv_smoothed > tl_bv_threshold).astype(int)
    # Detect rising edges only
    tl_bv_thresholded_diff = np.diff(tl_bv_thresholded)
    flip_times_bv_tl = np.squeeze(tl_time[np.where(tl_bv_thresholded_diff == 1)])
    raw_flip_times = {
        'tl_pd': _as_1d_float(flip_times_pd_tl),
        'tl_bv': _as_1d_float(flip_times_bv_tl),
        'bv': _as_1d_float(flip_times_bv_bv),
        'harp': _as_1d_float(flip_times_harp),
    }

    # in experiments with screens off there is no PD signal and thus no flips and 
    # can not therefore convert from TL time to Harp time. In these cases the
    # the encoder data is not retrievable from the harp data file in the usual way
    # and we thus use the BV encoder data
    if tl_pd_valid == True and harp_pd_valid == True:
        print('Both TL and Harp PD signals are valid')
        pd_valid = True
        harp_valid = True
    else:
        print('*** Warning: One or both of the PD signals are invalid. If this is an experiment with screens off this is expected. If not, there is a problem. ***')
        if debug:
            print('Debug mode enabled: continuing without prompting despite invalid PD signal.')
        else:
            choice = input("Do you want to continue? (y/n): ").strip().lower()
            if choice != 'y':
                print("Exiting...")
                return
        
        pd_valid = False
        harp_valid = True

    if filter_flips:
        min_pulses_unfiltered = min(len(flip_times_bv_bv),len(flip_times_pd_tl),len(flip_times_harp))
        raw_flip_times_pre_filter = {
            'tl_pd': _as_1d_float(flip_times_pd_tl),
            'tl_bv': _as_1d_float(flip_times_bv_tl),
            'bv': _as_1d_float(flip_times_bv_bv),
            'harp': _as_1d_float(flip_times_harp),
        }
        # before filtering check status
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Before filtering timing signal:')
        if len({len(flip_times_pd_tl),len(flip_times_harp),len(flip_times_bv_tl),len(flip_times_bv_bv)}) != 1:
            print('Number of flips detected in TL, BV and Harp do not match:')
            print('TL PD flips = ' + str(len(flip_times_pd_tl)))
            print('BV TL flips = ' + str(len(flip_times_bv_tl)))
            print('BV flips = ' + str(len(flip_times_bv_bv)))
            print('Harp flips = ' + str(len(flip_times_harp)))
        else:
            print('All timing pulses equal ('+str(len(flip_times_pd_tl))+')')
        
        pulse_time_diff_tl_bv_unfiltered = (flip_times_pd_tl[0:min_pulses_unfiltered]-flip_times_pd_tl[0])-(flip_times_bv_tl[0:min_pulses_unfiltered]-flip_times_bv_tl[0])
        # remove all pulses of < a certain width in tl/harp time

        # all_diff = np.diff(flip_times_pd_tl)
        # all_diff = all_diff[all_diff < 1]
        keep_idx_tl_pd = np.where((np.diff(flip_times_pd_tl) > min_pulse_width) & (np.diff(flip_times_pd_tl) < max_pulse_width))[0]
        keep_idx_harp = np.where((np.diff(flip_times_harp) > min_pulse_width) & (np.diff(flip_times_harp) < max_pulse_width))[0]
        keep_idx_tl_bv = np.where((np.diff(flip_times_bv_tl) > min_pulse_width) & (np.diff(flip_times_bv_tl) < max_pulse_width))[0]
        keep_idx_tl_pd = _append_last_index(keep_idx_tl_pd, len(flip_times_pd_tl))
        keep_idx_harp = _append_last_index(keep_idx_harp, len(flip_times_harp))
        keep_idx_tl_bv = _append_last_index(keep_idx_tl_bv, len(flip_times_bv_tl))
        flips_to_keep_bv = keep_idx_tl_bv

        flip_times_pd_tl_filtered = flip_times_pd_tl[keep_idx_tl_pd]
        flip_times_harp_filtered = flip_times_harp[keep_idx_harp]
        flip_times_dig_tl_filtered = flip_times_bv_tl[keep_idx_tl_bv]

        # flip_times_harp_filtered = flip_times_harp[np.where(np.diff(flip_times_harp) > min_width)[0]]
        # flip_times_dig_tl_filtered = flip_times_dig_tl[np.where(np.diff(flip_times_dig_tl) > min_width)[0]]
        # flips_to_keep_bv = np.where(np.diff(flip_times_dig_tl) > min_width)[0]
        
        flip_times_bv_bv = flip_times_bv_bv[flips_to_keep_bv]
        flip_times_harp = flip_times_harp_filtered
        flip_times_pd_tl = flip_times_pd_tl_filtered
        flip_times_bv_tl = flip_times_dig_tl_filtered

        # do sanity checks
        print('')
        print('After filtering timing signal:')
        if pd_valid:
            # number of flips should be the same on all systems if PD is valid
            if (len({len(flip_times_harp),len(flip_times_bv_bv)}) != 1) and (len({len(flip_times_pd_tl),len(flip_times_bv_tl),len(flip_times_bv_bv)}) == 1):
                # harp flip count is wrong but others are right so can still use BV data for encoder (caused by data frame issue)
                print('Number of flips detected in Harp and BV do not match after filtering both other timing pulses do match. You may thus continue by using rotary encoder data from BV log instead of harp:')
                print('Harp flips = ' + str(len(flip_times_harp)))
                print('BV flips = ' + str(len(flip_times_bv_bv)))
                print('This issue should not occur on data acquired after 25/03/2025 - please contact AR if you see this issue on data after this date.')
                if debug:
                    print('Debug mode enabled: continuing without prompting and switching to BV encoder data.')
                else:
                    choice = input("Do you want to continue? (y/n): ").strip().lower()
                    if choice != 'y':
                        print("Exiting...")
                        return
                # else set to not use PD and thus not use Harp for encoder
                pd_valid = True 
                harp_valid = False
            elif len({len(flip_times_pd_tl),len(flip_times_harp),len(flip_times_bv_tl),len(flip_times_bv_bv)}) != 1:
                print('Number of flips detected in TL, BV and Harp do not match:')
                print('TL PD flips = ' + str(len(flip_times_pd_tl)))
                print('BV TL flips = ' + str(len(flip_times_bv_tl)))
                print('BV flips = ' + str(len(flip_times_bv_bv)))
                print('Harp flips = ' + str(len(flip_times_harp)))
                raise ValueError('Pulse count mismatch')
            else:
                print('Number of flips detected in TL, BV and Harp match:')
                print('BV flips = ' + str(len(flip_times_bv_tl)))
            # the relative times of flips should be near identical between flip_times_pd_tl and flip_times_bv_tl
            pd_tl_v_bv_tl_jitter = np.abs((flip_times_pd_tl-flip_times_pd_tl[0]) - (flip_times_bv_tl-flip_times_bv_tl[0]))
            if max(pd_tl_v_bv_tl_jitter) > 50:
                print('Jitter between TL and BV timing pulses is too large:')
                print('Max jitter = ' + str(round(max(pd_tl_v_bv_tl_jitter)*1000)) + ' ms')
                raise ValueError('Jitter mismatch')
            else:
                print('Jitter between TL and BV timing pulses is acceptable:')
                print('Median jitter = ' + str(round(np.median(pd_tl_v_bv_tl_jitter)*1000))+ ' ms')
                print('Max jitter = ' + str(round(max(pd_tl_v_bv_tl_jitter)*1000)) + ' ms')
        else:
            # number of flips should be the same on BV and TL
            if len(flip_times_bv_bv) != len(flip_times_bv_tl):
                print('Number of flips detected in BV and TL do not match:')
                print('BV flips = ' + str(len(flip_times_bv_bv)))
                print('TL flips = ' + str(len(flip_times_bv_tl)))
                raise ValueError('Pulse count mismatch')
            else:
                print('Number of flips detected in BV and TL match:')
                print('BV flips = ' + str(len(flip_times_bv_bv)))
        print('Filtering complete')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    else:
        raw_flip_times_pre_filter = raw_flip_times.copy()
        keep_idx_tl_pd = np.arange(len(flip_times_pd_tl))
        keep_idx_harp = np.arange(len(flip_times_harp))
        keep_idx_tl_bv = np.arange(len(flip_times_bv_tl))
        flips_to_keep_bv = np.arange(len(flip_times_bv_bv))

    filtered_flip_times = {
        'tl_pd': _as_1d_float(flip_times_pd_tl),
        'tl_bv': _as_1d_float(flip_times_bv_tl),
        'bv': _as_1d_float(flip_times_bv_bv),
        'harp': _as_1d_float(flip_times_harp),
    }

    if debug:
        debug_defaults = {
            'time_range': None,
            'align_to_first_pulse': False,
            'excluded_window': 2.0,
        }
        if debug_params is None:
            debug_params = debug_defaults.copy()
        else:
            debug_params = {**debug_defaults, **debug_params}

        raw_guessed_flips, raw_reference = _guess_missing_flips_by_system(raw_flip_times)
        filtered_guessed_flips, filtered_reference = _guess_missing_flips_by_system(filtered_flip_times)

        print(f'Debug raw reference system for missing-flip guesses: {raw_reference}')
        print(f'Debug filtered reference system for missing-flip guesses: {filtered_reference}')

        excluded_flip_times = {
            'tl_pd': _find_excluded_flips(raw_flip_times['tl_pd'], filtered_flip_times['tl_pd']),
            'tl_bv': _find_excluded_flips(raw_flip_times['tl_bv'], filtered_flip_times['tl_bv']),
            'bv': _find_excluded_flips(raw_flip_times['bv'], filtered_flip_times['bv']),
            'harp': _find_excluded_flips(raw_flip_times['harp'], filtered_flip_times['harp']),
        }
        print(
            'Debug excluded flips by system: '
            f'tl_pd={len(excluded_flip_times["tl_pd"])}, '
            f'tl_bv={len(excluded_flip_times["tl_bv"])}, '
            f'bv={len(excluded_flip_times["bv"])}, '
            f'harp={len(excluded_flip_times["harp"])}'
        )
        excluded_flip_details = {
            'tl_pd': _build_excluded_details(
                raw_flip_times_pre_filter['tl_pd'], keep_idx_tl_pd, min_pulse_width, max_pulse_width,
                interval_source_flips=raw_flip_times_pre_filter['tl_pd'],
                reason_source_name='tl_pd',
                timeline_reference_flips=raw_flip_times_pre_filter['tl_pd'],
            ),
            'tl_bv': _build_excluded_details(
                raw_flip_times_pre_filter['tl_bv'], keep_idx_tl_bv, min_pulse_width, max_pulse_width,
                interval_source_flips=raw_flip_times_pre_filter['tl_bv'],
                reason_source_name='tl_bv',
                timeline_reference_flips=raw_flip_times_pre_filter['tl_bv'],
            ),
            'bv': _build_excluded_details(
                raw_flip_times_pre_filter['bv'], flips_to_keep_bv, min_pulse_width, max_pulse_width,
                interval_source_flips=raw_flip_times_pre_filter['tl_bv'],
                reason_source_name='tl_bv',
                timeline_reference_flips=raw_flip_times_pre_filter['tl_bv'],
            ),
            'harp': _build_excluded_details(
                raw_flip_times_pre_filter['harp'], keep_idx_harp, min_pulse_width, max_pulse_width,
                interval_source_flips=raw_flip_times_pre_filter['harp'],
                reason_source_name='harp',
                timeline_reference_flips=raw_flip_times_pre_filter['tl_pd'],
            ),
        }
        _print_excluded_details(excluded_flip_details)

        raw_panels = [
            _prepare_debug_panel(
                _make_debug_panel(
                    tl_time, tl_pd, raw_flip_times['tl_pd'], raw_guessed_flips['tl_pd'],
                    'Timeline photodiode signal (raw flips)', 'Timeline time (s)', system_name='tl_pd'
                ),
                debug_params,
            ),
            _prepare_debug_panel(
                _make_debug_panel(
                    tl_time, tl_bv, raw_flip_times['tl_bv'], raw_guessed_flips['tl_bv'],
                    'Timeline Bonvision signal (raw flips)', 'Timeline time (s)', system_name='tl_bv'
                ),
                debug_params,
            ),
            _prepare_debug_panel(
                _make_debug_panel(
                    Timestamp, Sync, raw_flip_times['bv'], raw_guessed_flips['bv'],
                    'Bonvision sync signal (raw flips)', 'BV time (s)', system_name='bv'
                ),
                debug_params,
            ),
            _prepare_debug_panel(
                _make_debug_panel(
                    harp_time, harp_pd, raw_flip_times['harp'], raw_guessed_flips['harp'],
                    'Harp photodiode signal (raw flips)', 'Harp time (s)', system_name='harp'
                ),
                debug_params,
            ),
        ]

        filtered_panels = [
            _prepare_debug_panel(
                _make_debug_panel(
                    tl_time, tl_pd, filtered_flip_times['tl_pd'], filtered_guessed_flips['tl_pd'],
                    'Timeline photodiode signal (filtered flips)', 'Timeline time (s)', system_name='tl_pd'
                ),
                debug_params,
            ),
            _prepare_debug_panel(
                _make_debug_panel(
                    tl_time, tl_bv, filtered_flip_times['tl_bv'], filtered_guessed_flips['tl_bv'],
                    'Timeline Bonvision signal (filtered flips)', 'Timeline time (s)', system_name='tl_bv'
                ),
                debug_params,
            ),
            _prepare_debug_panel(
                _make_debug_panel(
                    Timestamp, Sync, filtered_flip_times['bv'], filtered_guessed_flips['bv'],
                    'Bonvision sync signal (filtered flips)', 'BV time (s)', system_name='bv'
                ),
                debug_params,
            ),
            _prepare_debug_panel(
                _make_debug_panel(
                    harp_time, harp_pd, filtered_flip_times['harp'], filtered_guessed_flips['harp'],
                    'Harp photodiode signal (filtered flips)', 'Harp time (s)', system_name='harp'
                ),
                debug_params,
            ),
        ]

        excluded_panels = [
            _prepare_debug_panel(
                _make_debug_panel(
                    tl_time, tl_pd, raw_flip_times['tl_pd'], np.array([]),
                    'Timeline photodiode signal (excluded flip browser)', 'Timeline time (s)',
                    system_name='tl_pd',
                    filtered_flips=filtered_flip_times['tl_pd'],
                    excluded_flips=excluded_flip_times['tl_pd'],
                    excluded_details=excluded_flip_details['tl_pd'],
                ),
                debug_params,
                apply_time_range=False,
            ),
            _prepare_debug_panel(
                _make_debug_panel(
                    tl_time, tl_bv, raw_flip_times['tl_bv'], np.array([]),
                    'Timeline Bonvision signal (excluded flip browser)', 'Timeline time (s)',
                    system_name='tl_bv',
                    filtered_flips=filtered_flip_times['tl_bv'],
                    excluded_flips=excluded_flip_times['tl_bv'],
                    excluded_details=excluded_flip_details['tl_bv'],
                ),
                debug_params,
                apply_time_range=False,
            ),
            _prepare_debug_panel(
                _make_debug_panel(
                    Timestamp, Sync, raw_flip_times['bv'], np.array([]),
                    'Bonvision sync signal (excluded flip browser)', 'BV time (s)',
                    system_name='bv',
                    filtered_flips=filtered_flip_times['bv'],
                    excluded_flips=excluded_flip_times['bv'],
                    excluded_details=excluded_flip_details['bv'],
                ),
                debug_params,
                apply_time_range=False,
            ),
            _prepare_debug_panel(
                _make_debug_panel(
                    harp_time, harp_pd, raw_flip_times['harp'], np.array([]),
                    'Harp photodiode signal (excluded flip browser)', 'Harp time (s)',
                    system_name='harp',
                    filtered_flips=filtered_flip_times['harp'],
                    excluded_flips=excluded_flip_times['harp'],
                    excluded_details=excluded_flip_details['harp'],
                ),
                debug_params,
                apply_time_range=False,
            ),
        ]

        _show_debug_tabs(
            expID,
            raw_panels,
            filtered_panels,
            excluded_panels=excluded_panels,
            excluded_window=debug_params['excluded_window'],
        )
        _show_pulse_diagnostics(raw_flip_times, filtered_flip_times, min_pulse_width, max_pulse_width)


    # Use the BV ground truth for the number of flips that should be present
    true_flips = len(flip_times_bv_bv)

    # Check that the number of flips detected in the PD signal is the same as the number of flips detected in the BV signal

    # Fit model to convert BV time to TL time either using PD or digital flip signal from BV
    if pd_valid:
        # use PD
        interp_bv_flips, interp_tl_flips = _build_interp_pair(
            flip_times_bv_bv[0:true_flips],
            flip_times_pd_tl[0:true_flips],
            debug=debug,
            pair_name='BV/PD flip pair',
        )
        linear_interpolator_bv_2_tl = interp1d(interp_bv_flips, interp_tl_flips, kind='linear', fill_value="extrapolate")
    else:
        # use digital signal
        interp_bv_flips, interp_tl_flips = _build_interp_pair(
            flip_times_bv_bv[0:true_flips],
            flip_times_bv_tl[0:true_flips],
            debug=debug,
            pair_name='BV/TL digital flip pair',
        )
        linear_interpolator_bv_2_tl = interp1d(interp_bv_flips, interp_tl_flips, kind='linear', fill_value="extrapolate")
                                               
    if pd_valid:
        # Fit model to convert Harp time to TL time
        interp_harp_flips, interp_pd_flips = _build_interp_pair(
            flip_times_harp[0:true_flips],
            flip_times_pd_tl[0:true_flips],
            debug=debug,
            pair_name='Harp/PD flip pair',
        )
        linear_interpolator_harp_2_tl = interp1d(interp_harp_flips, interp_pd_flips, kind='linear', fill_value="extrapolate")

        # Check all systems registered the same number of pulses
        if len(flip_times_harp) == len(flip_times_bv_bv) == len(flip_times_pd_tl):
            print ('Pulse count matches accross TL/BV/Harp')
            print ('Pulse count = ' + str(len(flip_times_harp)))
        elif len(flip_times_bv_bv) == len(flip_times_pd_tl):
            print ('Pulse count matches accross TL/BV BUT NOT HARP')
            print ('Pulse count = ' + str(len(flip_times_harp)))    
            print('!!!Please flag this up with Adam!!!')        
        else:
            print('Harp pulses = ' + str(len(flip_times_harp)))
            print('BV pulses = ' + str(len(flip_times_bv_bv)))
            print('TL pulses = ' + str(len(flip_times_pd_tl)))
            raise ValueError('Pulse count mismatch')
    else:
        # Check all systems registered the same number of pulses
        if len(flip_times_bv_bv) == len(flip_times_bv_tl):
            print ('Pulse count matches accross TL/BV')
            print ('Pulse count = ' + str(len(flip_times_bv_bv)))
        else:
            print('BV pulses = ' + str(len(flip_times_bv_bv)))
            print('TL pulses = ' + str(len(flip_times_bv_tl)))
            raise ValueError('Pulse count mismatch')        
    
    # get trial onset times
    # in BV time
    # find the moments when the (bonsai) trial number increments
    trialOnsetTimesBV = Timestamp[np.where(np.diff(Trial)==1)]
    # add in first trial onset
    trialOnsetTimesBV = np.insert(trialOnsetTimesBV,0,Timestamp[0])
    # in TL time
    trialOnsetTimesBV_tl = linear_interpolator_bv_2_tl(trialOnsetTimesBV)

    # load matlab expData file
    expData = loadmat(os.path.join(exp_dir_raw, expID + '_stim.mat'))
    stims = expData['expDat']['stims']
    stims = stims[0][0][0]

    stim_info = pd.read_csv(os.path.join(exp_dir_raw, expID + '_stim.csv'))
    stim_order = pd.read_csv(os.path.join(exp_dir_raw, expID + '_stim_order.csv'), header=None)

    # make a matrix for csv output of trial onset time and trial stimulus type
    # check number of trial onsets matches between bonvision and bGUI
    if len(trialOnsetTimesBV_tl) != stim_order.shape[0]:
        raise ValueError(
            'Number of trial onsets doesn\'t match between bonvision and bGUI - there is a likely logging issue')
    else:
        print('Number of trial onsets matches between bonvision and bGUI')
        print('Number of trials = ' + str(len(trialOnsetTimesBV_tl)))
    # make the matrix of trial onset times
    trialTimeMatrix = np.column_stack((trialOnsetTimesBV_tl, stim_order.values))

    # Add running trace
    if pd_valid and harp_valid:
        # then we can use harp for encoder - before scaling this is raw ticks of the rotary encoder
        wheel_pos = harp_encoder
        wheel_timestamps = linear_interpolator_harp_2_tl(harp_time)
        wheel_pos = wheel_pos * encoder_scaling_factor
    else:
        # we use BV for encoder - this is scaled using scale factor in bonsai
        wheel_pos = bv_encoder['Encoder'].values
        wheel_timestamps = linear_interpolator_bv_2_tl(bv_encoder['Timestamp'].values)

    # deal with wrap around of rotary encoder position
    wheel_pos_dif = np.diff(wheel_pos)
    wheel_pos_dif[wheel_pos_dif > 50000] -= 2**16
    wheel_pos_dif[wheel_pos_dif < -50000] += 2**16
    wheel_pos = np.cumsum(wheel_pos_dif)
    wheel_pos = np.append(wheel_pos,wheel_pos[-1])
    
    # Resample wheel to 20Hz
    resample_freq = 20
    wheel_linear_timescale = np.arange(0, np.floor(wheel_timestamps[-1]), 1/resample_freq)
    # Create the interpolater for resampling
    wheel_resampler = interpolate.interp1d(wheel_timestamps, wheel_pos, kind='linear',fill_value=(wheel_pos[0], wheel_pos[-1]), bounds_error=False)
    # Infer the wheel pos at each point on linear timescale 
    wheel_pos_resampled = wheel_resampler(wheel_linear_timescale)
    # smooth this position data to deal with the rotary encoder encoding discrete steps
    smooth_window = 10 # window size for smoothing (10 = 0.5 secs)
    wheel_pos_smooth = np.convolve(wheel_pos_resampled, np.ones(smooth_window)/smooth_window, mode='same')
    # set smooth_window at start and end to the first and last value of the unsmoothed data
    wheel_pos_smooth[0:smooth_window] = wheel_pos_resampled[0]
    wheel_pos_smooth[-smooth_window:] = wheel_pos_resampled[-1]
    # Calc diff between position samples (already in units oyf meters)
    # mouse velocity in cm/sample (at 20Hz)
    wheel_velocity = np.diff(wheel_pos_smooth) 
    wheel_velocity = np.append(wheel_velocity, wheel_velocity[-1])
    # mouse velocity in m/s
    wheel_velocity = wheel_velocity * resample_freq
    # Save data
    wheel = {}
    wheel['position'] = np.array(wheel_pos_resampled)
    wheel['position_smoothed'] = np.array(wheel_pos_smooth)
    wheel['speed'] = np.array(wheel_velocity)
    wheel['t'] = np.array(wheel_linear_timescale)
    if not debug:
        with open(os.path.join(exp_dir_processed_recordings, 'wheel.pickle'), 'wb') as f:
            pickle.dump(wheel, f)

    # output a csv file which contains dataframe of all trials with first column showing trial onset time
    # read the all trials file, append trial onset times to first column (trialOnsetTimesTL)
    if not debug:
        all_trials = pd.read_csv(os.path.join(exp_dir_raw, expID + '_all_trials.csv'))
        all_trials.insert(0,'time',trialOnsetTimesBV_tl)
        all_trials.to_csv(os.path.join(exp_dir_processed, expID + '_all_trials.csv'), index=False)
        print('Done without errors')
    else:
        print('Debug mode enabled: no files were saved')

    # for debugging:
def main():
    # userID = 'melinatimplalexi'
    userID = 'yannickbollmann'
    #userID = 'adamranson'
    expID = '2026-03-24_08_ESYB035'
    run_preprocess_bv2(
        userID,
        expID,
        debug=True,
        debug_params={'align_to_first_pulse': True},
    )

    # debug_params={'time_range': (-10, 500), 'align_to_first_pulse': True},


if __name__ == "__main__":
    main()
