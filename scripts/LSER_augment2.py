import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import find_peaks, peak_widths


def debug_find_peaks_in_range(spectrum, wavenumbers, wn_start, wn_end, min_prominence_ratio=0.05):
    #print(f"\n🔍 DEBUG: Looking for dominant peak in range {wn_start}-{wn_end} cm⁻¹")

    if wn_start > wn_end:
        wn_start, wn_end = wn_end, wn_start
    mask = (wavenumbers >= wn_start) & (wavenumbers <= wn_end)
    region_indices = np.where(mask)[0]

    if len(region_indices) < 3:
        #print("❌ Not enough points for peak detection")
        return []

    region_spectrum = spectrum[region_indices]

    # Detect peak direction
    peak_type = 'down' if np.median(region_spectrum) > np.mean(spectrum) else 'up'
    #print(f"🔁 Detected peak type: {'↓ minima' if peak_type == 'down' else '↑ maxima'}")

    if peak_type == 'down':
        processed_spectrum = 1.0 - spectrum
    else:
        processed_spectrum = spectrum

    local_range = np.max(region_spectrum) - np.min(region_spectrum)
    if local_range == 0:
        local_range = 1.0
    dynamic_prominence = max(min_prominence_ratio * local_range, 1e-4)

    peak_indices, _ = find_peaks(
        processed_spectrum[region_indices],
        distance=1,
        prominence=dynamic_prominence
    )

    if len(peak_indices) == 0:
        fallback_idx = np.argmin(region_spectrum) if peak_type == 'down' else np.argmax(region_spectrum)
        global_peak_idx = region_indices[fallback_idx]
        peak_wn = wavenumbers[global_peak_idx]
        peak_intensity = spectrum[global_peak_idx]
        left_bound = max(0, global_peak_idx - 3)
        right_bound = min(len(spectrum) - 1, global_peak_idx + 3)
        width = abs(wavenumbers[left_bound] - wavenumbers[right_bound])
        #print(f"⚠️ Using fallback for peak at {peak_wn:.1f} cm⁻¹ (T={peak_intensity:.3f})")
        return [{
            'peak_idx': global_peak_idx,
            'peak_wn': peak_wn,
            'peak_intensity': peak_intensity,
            'left_bound': left_bound,
            'right_bound': right_bound,
            'width': width,
            'type': peak_type
        }]

    if peak_type == 'down':
        best_idx = np.argmin(spectrum[region_indices[peak_indices]])
    else:
        best_idx = np.argmax(spectrum[region_indices[peak_indices]])

    global_peak_idx = region_indices[peak_indices[best_idx]]
    peak_wn = wavenumbers[global_peak_idx]
    peak_intensity = spectrum[global_peak_idx]

    widths, _, left_ips, right_ips = peak_widths(processed_spectrum, [global_peak_idx], rel_height=0.95)
    left_bound = max(0, int(np.round(left_ips[0])))
    right_bound = min(len(spectrum) - 1, int(np.round(right_ips[0])))
    width = abs(wavenumbers[left_bound] - wavenumbers[right_bound])

    # print(f"✅ Found peak at {peak_wn:.1f} cm⁻¹ (T={peak_intensity:.3f}), width ≈ {width:.1f} cm⁻¹")

    return [{
        'peak_idx': global_peak_idx,
        'peak_wn': peak_wn,
        'peak_intensity': peak_intensity,
        'left_bound': left_bound,
        'right_bound': right_bound,
        'width': width,
        'type': peak_type
    }]

from scipy.ndimage import gaussian_filter1d


def debug_apply_lser_shifts(spectrum, wavenumbers, functional_groups, pi_star,beta,alpha):
    # print(f"\n🧪 DEBUG: Applying LSER shifts with π* = {pi_star}")
    
    lser_sensitivity = {
        'alcohols': -60.3,
        'carboxylic acids': -60.3,
        'ketones': -14.2,
        'aldehydes': -14.2,
        'esters': -16.6,
        'amides': -22.8,
        'nitriles': -20.0,
        'alkyl halides': -16.8,
        'nitro': -25.2,
    }

    #b
    lser_sensitivity2 = {
        'alcohols': -288.99,
        'carboxylic acids': 0.00,
        'ketones': 0.46, 
        'aldehydes': 0.46, 
        'esters': 1.67, 
        'amides': -14.96,
        'nitriles': 0.00,
        'alkyl halides': 0.00, 
        'nitro': 0.00,
    }
    #a
    lser_sensitivity3 = {
        'alcohols': 94.46, # done
        'carboxylic acids': 0.00,
        'ketones': -12.3, # done
        'aldehydes': -12.3, # done
        'esters': -20.16, #done
        'amides': -25.075, # done
        'nitriles': 0.00,
        'alkyl halides': 0.00, #done
        'nitro': 0.00,
    }


    functional_group_ranges = {
        'alcohols': [(3200, 3700)],
        'carboxylic acids': [(2500, 2800), (1600, 1900), (1300, 1000)],
        'ketones': [(1600, 1900)],
        'aldehydes': [(1600, 1900)],
        'esters': [(1600, 1900), (1500, 1000)],
        'amides': [(1600, 1900)],
        'nitriles': [(2100, 2300)],
        'alkyl halides': [(500, 850)],
        'nitro': [(1300, 1500), (1500, 1800)],
    }

    shifted_spectrum = spectrum.copy()
    temp_spectrum = shifted_spectrum.copy()
    total_peaks_shifted = 0
    
    # Calculate baseline threshold (1.05x median of entire spectrum)
    
    spectrum_median = np.median(spectrum)

    spectrum_min = np.min(spectrum)
    baseline_threshold = 0.03*np.mean(spectrum)
    #print(f"📊 Spectrum median: {spectrum_min:.4f}")
    #print(f"📊 Baseline threshold (1.05×): {baseline_threshold:.4f}")

    for group_name, is_present in functional_groups.items():
        if not is_present or group_name not in lser_sensitivity:
            continue

        s_parameter = lser_sensitivity[group_name]
        b_parameter = lser_sensitivity2[group_name]
        a_parameter = lser_sensitivity3[group_name]
        frequency_shift = s_parameter * pi_star  + b_parameter * beta + a_parameter *alpha




        if abs(frequency_shift) < 1.0:
            continue

        # print(f"\n📍 Functional group: {group_name}")
        # print(f"   Shift: {frequency_shift:.1f} cm⁻¹")

        for wn_start, wn_end in functional_group_ranges.get(group_name, []):
            peaks = debug_find_peaks_in_range(shifted_spectrum, wavenumbers, wn_start, wn_end)
            if not peaks:
                continue

            # Only take the most prominent peak
            peak_info = peaks[0]
            current_wn = peak_info['peak_wn']
            new_peak_wn = current_wn + frequency_shift
            peak_idx = peak_info['peak_idx']
            peak_intensity = peak_info['peak_intensity']

            # Start with the original peak width as initial search zone
            peak_width_pts = peak_info['right_bound'] - peak_info['left_bound']
            initial_pad = int(peak_width_pts * 2.2)
            
            # Define initial search boundaries
            search_lb = max(0, peak_idx - initial_pad)
            search_rb = min(len(spectrum) - 1, peak_idx + initial_pad)
            
            # print(f"   🔍 Peak at index {peak_idx} ({current_wn:.1f} cm⁻¹)")
            # print(f"   🔍 Initial search zone: indices {search_lb}-{search_rb}")
            
            # Find left boundary: search leftward from peak for baseline
            left_bound = peak_idx
            for i in range(peak_idx - 1, search_lb - 1, -1):
                if spectrum[i] <= baseline_threshold:
                    left_bound = i
                    break
            else:
                left_bound = search_lb  # Fallback if threshold not found
            
            # Find right boundary: search rightward from peak for baseline  
            right_bound = peak_idx
            for i in range(peak_idx + 1, search_rb + 1):
                if spectrum[i] <= baseline_threshold:
                    right_bound = i
                    break
            else:
                right_bound = search_rb  # Fallback if threshold not found
            
            #print(f"   📏 Dynamic boundaries: indices {left_bound}-{right_bound}")
            #print(f"   📏 Wavenumber range: {wavenumbers[left_bound]:.1f}-{wavenumbers[right_bound]:.1f} cm⁻¹")
            #print(f"   📏 Intensity at boundaries: L={spectrum[left_bound]:.4f}, R={spectrum[right_bound]:.4f}")
            
            # Extract the dynamically determined region
            region_wn = wavenumbers[left_bound:right_bound + 1]
            region_intensity = spectrum[left_bound:right_bound + 1]

            target_wn = region_wn + frequency_shift
            mask = (wavenumbers >= target_wn.min()) & (wavenumbers <= target_wn.max())
            target_indices = np.where(mask)[0]

            if len(target_indices) >= 3:
                sort_idx = np.argsort(target_wn)
                x_sorted = target_wn[sort_idx]
                y_sorted = region_intensity[sort_idx]

                if len(x_sorted) > 5:
                    interp = UnivariateSpline(x_sorted, y_sorted, s=0, k=3)
                else:
                    interp = interp1d(x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')

                # Flatten original region in buffer
                temp_spectrum[left_bound:right_bound + 1] = spectrum_median

                # Insert interpolated shifted peak
                temp_spectrum[target_indices] = interp(wavenumbers[target_indices])
                total_peaks_shifted += 1

                # print(f"   🔄 Shifted {peak_info['type']} peak region {wavenumbers[left_bound]:.1f}-{wavenumbers[right_bound]:.1f} → {new_peak_wn:.1f} cm⁻¹")

    # Final smoothing
    shifted_spectrum = gaussian_filter1d(temp_spectrum, sigma=1)

    return shifted_spectrum, total_peaks_shifted

# === Load and process data ===
"""
jdx_path = r"C:/Users/krasz/Downloads/67-56-1-IR.jdx"
jdx_data = jcamp_readfile(jdx_path)
wavenumbers = np.array(jdx_data['x'])
spectrum = np.array(jdx_data['y'])

print("🔬 LSER Debug Analysis")
print("=" * 50)
print(f"Spectrum length: {len(spectrum)} points")
print(f"Wavenumber range: {wavenumbers[0]:.0f} to {wavenumbers[-1]:.0f} cm⁻¹")
print(f"Transmittance range: {np.min(spectrum):.3f} to {np.max(spectrum):.3f}")

functional_groups = {
    'alcohols': True,
    'ketones': False,
    'aldehydes': False,
    'esters': False,
    'amides': False,
    'nitriles': False,
    'carboxylic acids': False,
    'alkyl halides': False,
    'nitro': False
}

pi_star = 0.71  # e.g., acetone

shifted_spectrum, peaks_shifted = debug_apply_lser_shifts(spectrum, wavenumbers, functional_groups, pi_star)

plt.figure(figsize=(10, 6))
plt.plot(wavenumbers, spectrum, label='Original Spectrum', color='black', linewidth=1.5)
plt.plot(wavenumbers, shifted_spectrum, label='Shifted Spectrum', color='red', linestyle='--', linewidth=1.5)
plt.gca().invert_xaxis()
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Transmittance')
plt.title('Original vs Shifted Spectrum')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n📊 Final Results:")
print(f"Total peaks shifted: {peaks_shifted}")
"""