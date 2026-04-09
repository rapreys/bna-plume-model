# ============================================================
# GAUSSIAN PLUME MODEL - BNA AIRPORT Emissions (Streamlit app)
# ============================================================
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import contextily as cx
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.dpi': 120, 'font.size': 11,
                     'axes.titlesize': 13, 'axes.labelsize': 11})

st.set_page_config(page_title="BNA Plume Model", layout="wide")
st.title("Gaussian Plume Model — BNA Airport Emissions")
st.caption("Screening-level dispersion model for emissions from Nashville International Airport (BNA).")

# ------------------------------------------------------------
# SIDEBAR: USER INPUTS
# ------------------------------------------------------------
with st.sidebar:
    st.header("Model Inputs")

    Q_g_s = st.number_input(
        "Emission rate (g/s)",
        min_value=0.0, max_value=5000.0, value=93.7, step=1.0, format="%.4f",
        help="Emission rate in grams per second. "
             "For reference: 1 metric ton/day ≈ 11.57 g/s."
    )

    STABILITY = st.selectbox(
        "Atmospheric stability class",
        options=['A', 'B', 'C', 'D', 'E', 'F'],
        index=3,
        help=("A = very unstable (strong daytime sun, light winds)\n"
              "B = moderately unstable\n"
              "C = slightly unstable\n"
              "D = neutral (overcast or moderate wind — most common)\n"
              "E = slightly stable (night, partial cloud)\n"
              "F = stable (clear night, light winds)")
    )

    WIND_DIR_DEG = st.slider(
        "Wind direction (degrees, FROM)",
        min_value=0, max_value=360, value=180, step=5,
        help=("Meteorological convention: direction wind is COMING FROM.\n"
              "0/360 = N, 90 = E, 180 = S (Nashville prevailing), 270 = W")
    )

    U_WIND = st.slider(
        "Wind speed (m/s)",
        min_value=0.5, max_value=15.0, value=5.0, step=0.1,
        help="Stability D typically requires wind ≥ 5 m/s for internal consistency."
    )

    EMISSION_TYPE = st.text_input(
        "Emission type label",
        value="CO",
        help="Label used in plot titles and axis labels (e.g. NOx, CO, PM2.5)."
    )

    NAAQS_VALUE = st.number_input(
        "NAAQS reference (µg/m³, 0 = none)",
        min_value=0.0, max_value=50000.0, value=0.0, step=1.0,
        help=("Optional reference line on the centreline plot.\n"
              "Examples: NO2 annual=100, NO2 1-hr=188, "
              "PM2.5 annual=9, PM2.5 24-hr=35, CO 1-hr=40000")
    )
    NAAQS_VALUE = NAAQS_VALUE if NAAQS_VALUE > 0 else None
    NAAQS_LABEL = f"NAAQS reference ({NAAQS_VALUE} µg/m³)" if NAAQS_VALUE else None

    CROSSWIND_DIST_KM = st.slider(
        "Crosswind profile distance (km)",
        min_value=0.5, max_value=20.0, value=5.0, step=0.5,
        help="Downwind distance at which to draw the crosswind concentration profile."
    )

    MAP_EXTENT_DEG = st.slider(
        "Zoom",
        min_value=0.04, max_value=0.30, value=0.08, step=0.01,
        help="Smaller = more zoomed in. Try 0.08–0.10 for low emissions, 0.20 for high."
    )

    st.markdown("---")
    st.header("Box Exposure")

    north = st.number_input(
        "North latitude", min_value=35.9, max_value=36.4,
        value=36.2226, step=0.001, format="%.4f"
    )
    south = st.number_input(
        "South latitude", min_value=35.9, max_value=36.4,
        value=36.1473, step=0.001, format="%.4f"
    )
    east = st.number_input(
        "East longitude", min_value=-86.9, max_value=-86.4,
        value=-86.6394, step=0.001, format="%.4f"
    )
    west = st.number_input(
        "West longitude", min_value=-86.9, max_value=-86.4,
        value=-86.7096, step=0.001, format="%.4f"
    )

    exposure_hours = st.number_input(
        "Exposure duration (hours)",
        min_value=0.1, max_value=24.0, value=1.0, step=0.1,
        help="Time window for integrated exposure calculation."
    )

    population = st.number_input(
        "Population in box",
        min_value=0, max_value=1_000_000, value=20000, step=1000,
        help="Assumed number of people uniformly in the box (screening only)."
    )

# ------------------------------------------------------------
# MODEL CODE
# ------------------------------------------------------------
BNA_LAT = 36.1245
BNA_LON = -86.6782
M_PER_DEG_LAT = 111_320
M_PER_DEG_LON = 111_320 * np.cos(np.radians(BNA_LAT))

PG = {
    'A': {'a': 0.3658, 'b': 0.9031, 'c': 0.192,  'd': 1.0857},
    'B': {'a': 0.2751, 'b': 0.9031, 'c': 0.156,  'd': 0.9644},
    'C': {'a': 0.2090, 'b': 0.9031, 'c': 0.116,  'd': 0.9031},
    'D': {'a': 0.1471, 'b': 0.9031, 'c': 0.079,  'd': 0.8804},
    'E': {'a': 0.1046, 'b': 0.9031, 'c': 0.063,  'd': 0.8530},
    'F': {'a': 0.0722, 'b': 0.9031, 'c': 0.053,  'd': 0.8302},
}

def sigma_y(x, stab):
    c = PG[stab]
    return c['a'] * np.power(np.maximum(x, 1.0), c['b'])

def sigma_z(x, stab):
    c = PG[stab]
    return c['c'] * np.power(np.maximum(x, 1.0), c['d'])

def gaussian_plume(x, y, Q, u, stab):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sy = sigma_y(x, stab)
    sz = sigma_z(x, stab)
    C = np.zeros_like(x)
    v = x > 0
    C[v] = (Q / (np.pi * u * sy[v] * sz[v])) * np.exp(-0.5 * (y[v] / sy[v])**2)
    return C

def lonlat_to_local(lon, lat, wind_deg):
    east_m  = (lon - BNA_LON) * M_PER_DEG_LON
    north_m = (lat - BNA_LAT) * M_PER_DEG_LAT
    theta = np.radians((wind_deg + 180) % 360)
    x = east_m * np.sin(theta) + north_m * np.cos(theta)
    y = east_m * np.cos(theta) - north_m * np.sin(theta)
    return x, y

def smart_contour_levels(cmin, cmax, n=6):
    candidates = []
    for exp in range(math.floor(math.log10(cmin)), math.ceil(math.log10(cmax)) + 1):
        for m in [1, 2, 5]:
            v = m * 10**exp
            if cmin < v < cmax:
                candidates.append(v)
    if not candidates:
        return []
    step = max(1, len(candidates) // n)
    return candidates[::step]

def fmt_conc(v):
    if v >= 100:
        return f'{v:.1f}'
    elif v >= 0.01:
        return f'{v:.4f}'
    else:
        return f'{v:.3e}'

dirs = ['N','NE','E','SE','S','SW','W','NW','N']
plume_dir = dirs[round(((WIND_DIR_DEG + 180) % 360) / 45)]

# For display convenience
Q_tons_day = Q_g_s * 86400 / 1e6

# ------------------------------------------------------------
# PLOT 1: MAP HEATMAP
# ------------------------------------------------------------
st.subheader("1. Ground-level concentration map")

lon_1d = np.linspace(BNA_LON - MAP_EXTENT_DEG, BNA_LON + MAP_EXTENT_DEG, 500)
lat_1d = np.linspace(BNA_LAT - MAP_EXTENT_DEG * 0.9, BNA_LAT + MAP_EXTENT_DEG * 0.9, 400)
LON, LAT = np.meshgrid(lon_1d, lat_1d)

Xl, Yl = lonlat_to_local(LON, LAT, WIND_DIR_DEG)
C_ugm3 = gaussian_plume(Xl, Yl, Q_g_s, U_WIND, STABILITY) * 1e6

# Grid-INDEPENDENT colorbar reference: analytical centerline concentration
# at a fixed reference distance, so the colorbar doesn't jump when the plume rotates.
ref_dist = 200.0  # m
cmax = (Q_g_s / (np.pi * U_WIND
                 * sigma_y(ref_dist, STABILITY)
                 * sigma_z(ref_dist, STABILITY))) * 1e6
floor = max(1e-8, cmax * 1e-4)

C_plot = np.ma.masked_less_equal(C_ugm3, floor)
pos    = C_ugm3[C_ugm3 > floor]

if len(pos) > 0:
    cmin = pos.min()
    levels = np.geomspace(cmin, cmax, 30)
    norm   = mcolors.LogNorm(vmin=cmin, vmax=cmax)

    fig1, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(lon_1d.min(), lon_1d.max())
    ax.set_ylim(lat_1d.min(), lat_1d.max())
    ax.set_aspect(1.0 / np.cos(np.radians(BNA_LAT)))
    try:
        cx.add_basemap(ax, crs='EPSG:4326',
                       source=cx.providers.OpenStreetMap.Mapnik, zoom=12)
    except Exception as e:
        st.warning(f"Basemap could not load ({e}). Showing plume without map background.")

    cf = ax.contourf(LON, LAT, C_plot, levels=levels,
                     norm=norm, cmap='YlOrRd', alpha=0.55, zorder=2)

    line_lvls = smart_contour_levels(cmin, cmax, n=6)
    if line_lvls:
        cs = ax.contour(LON, LAT, C_plot, levels=line_lvls,
                        colors='k', linewidths=0.7, linestyles='--',
                        alpha=0.8, zorder=3)
        ax.clabel(cs, fmt=lambda v: f'{fmt_conc(v)} µg/m³', fontsize=7)

    cbar = plt.colorbar(cf, ax=ax,
                        label=f'{EMISSION_TYPE} concentration (µg/m³)',
                        shrink=0.75, pad=0.02)
    cbar.ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=20))
    cbar.ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False))
    cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())

    # Overlay the exposure box (if it's a valid box)
    if (north > south) and (east > west):
        box_lons = [west, east, east, west, west]
        box_lats = [south, south, north, north, south]
        ax.plot(box_lons, box_lats, color='blue', linewidth=2, ls='-',
                label='Exposure box', zorder=4)

    ax.plot(BNA_LON, BNA_LAT, 'w^', ms=16,
            markeredgecolor='k', markeredgewidth=2,
            zorder=5, label='BNA Airport')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(
        f'Ground-level {EMISSION_TYPE} Plume over Nashville\n'
        f'Q = {Q_g_s:.2f} g/s ({Q_tons_day:.3f} MT/day)  |  u = {U_WIND} m/s  |  '
        f'Stability {STABILITY}  |  Wind from {WIND_DIR_DEG}° → plume to {plume_dir}'
    )
    ax.legend(loc='upper right', fontsize=10,
              facecolor='white', edgecolor='grey', framealpha=0.9)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
else:
    st.warning("Concentrations too low to plot at this scale. Try increasing emission rate or zooming in.")

# ------------------------------------------------------------
# PLOT 2: CENTRELINE CONCENTRATION vs DOWNWIND DISTANCE
# ------------------------------------------------------------
st.subheader("2. Centreline concentration vs downwind distance")

x_down   = np.linspace(100, 20000, 600)
C_centre = gaussian_plume(x_down, np.zeros_like(x_down),
                          Q_g_s, U_WIND, STABILITY) * 1e6

fig2, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_down / 1000, C_centre, color='crimson', lw=2,
        label=f'Centreline {EMISSION_TYPE}')
ax.axvline(CROSSWIND_DIST_KM, color='royalblue', lw=1.5, ls=':',
           label=f'Crosswind profile @ {CROSSWIND_DIST_KM} km')

c_plot_max = C_centre.max()
c_plot_min = C_centre.min()
if NAAQS_VALUE is not None and c_plot_min <= NAAQS_VALUE <= c_plot_max * 5:
    ax.axhline(NAAQS_VALUE, color='darkorange', lw=1.4, ls='--',
               label=NAAQS_LABEL)
elif NAAQS_VALUE is not None:
    st.info(f'Note: NAAQS reference ({NAAQS_VALUE} µg/m³) is outside the plotted '
            f'range ({fmt_conc(c_plot_min)} – {fmt_conc(c_plot_max)} µg/m³) '
            f'and was not drawn.')

ax.set_yscale('log')
ax.set_xlabel('Downwind distance (km)')
ax.set_ylabel(f'Centreline {EMISSION_TYPE} concentration (µg/m³)')
ax.set_title(
    f'Centreline {EMISSION_TYPE} Concentration vs Downwind Distance\n'
    f'Q = {Q_g_s:.2f} g/s  |  u = {U_WIND} m/s  |  Stability {STABILITY}'
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ------------------------------------------------------------
# PLOT 3: CROSSWIND PROFILE
# ------------------------------------------------------------
st.subheader("3. Crosswind concentration profile")

x_cw = CROSSWIND_DIST_KM * 1000
y_cw = np.linspace(-8000, 8000, 400)
C_cw = gaussian_plume(np.full_like(y_cw, x_cw), y_cw,
                      Q_g_s, U_WIND, STABILITY) * 1e6

peak  = C_cw.max()
half  = peak / 2
above = y_cw[C_cw >= half]
hw    = (above[-1] - above[0]) / 2 / 1000 if len(above) > 1 else float('nan')

fig3, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_cw / 1000, C_cw, color='steelblue', lw=2)
ax.fill_between(y_cw / 1000, C_cw, alpha=0.15, color='steelblue')
ax.axhline(half, color='grey', lw=1.2, ls='--',
           label=f'Half-peak ({fmt_conc(half)} µg/m³)  |  half-width = {hw:.2f} km')
ax.set_xlabel('Crosswind distance from centreline (km)')
ax.set_ylabel(f'{EMISSION_TYPE} concentration (µg/m³)')
ax.set_title(
    f'Crosswind Profile at {CROSSWIND_DIST_KM} km Downwind\n'
    f'Peak = {fmt_conc(peak)} µg/m³  |  Half-width ≈ {hw:.2f} km'
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# ------------------------------------------------------------
# PLOT 4 + BOX EXPOSURE CALCULATION
# ------------------------------------------------------------
st.subheader("4. Box exposure")

if not ((north > south) and (east > west)):
    st.error("Invalid box: North must be greater than South, and East must be greater than West.")
else:
    nx_box, ny_box = 150, 150
    lon_box = np.linspace(west, east, nx_box)
    lat_box = np.linspace(south, north, ny_box)
    LON_box, LAT_box = np.meshgrid(lon_box, lat_box)

    X_box, Y_box = lonlat_to_local(LON_box, LAT_box, WIND_DIR_DEG)
    C_box_ugm3 = gaussian_plume(X_box, Y_box, Q_g_s, U_WIND, STABILITY) * 1e6

    # Area
    width_m  = (east - west) * M_PER_DEG_LON
    height_m = (north - south) * M_PER_DEG_LAT
    area_m2  = abs(width_m * height_m)
    area_km2 = area_m2 / 1e6

    # Stats
    mean_conc = float(np.mean(C_box_ugm3))
    max_conc  = float(np.max(C_box_ugm3))
    min_conc  = float(np.min(C_box_ugm3))

    # Exposure
    mean_exposure = mean_conc * exposure_hours
    max_exposure  = max_conc  * exposure_hours
    population_exposure = mean_exposure * population

    # --- Box map plot ---
    fig4, ax = plt.subplots(figsize=(10, 8))
    pad = 0.01
    ax.set_xlim(west - pad, east + pad)
    ax.set_ylim(south - pad, north + pad)
    ax.set_aspect(1.0 / np.cos(np.radians(BNA_LAT)))
    try:
        cx.add_basemap(ax, crs='EPSG:4326',
                       source=cx.providers.OpenStreetMap.Mapnik, zoom=12)
    except Exception as e:
        st.warning(f"Basemap could not load ({e}).")

    positive_vals = C_box_ugm3[C_box_ugm3 > 0]
    if len(positive_vals) > 0:
        # Reuse the grid-independent cmax from the main map so the two color
        # scales are comparable.
        cmin_box = max(positive_vals.min(), cmax * 1e-4)
        if cmin_box < cmax:
            levels_box = np.geomspace(cmin_box, cmax, 25)
            norm_box = mcolors.LogNorm(vmin=cmin_box, vmax=cmax)
            cf_box = ax.contourf(LON_box, LAT_box, C_box_ugm3,
                                 levels=levels_box, norm=norm_box,
                                 cmap='YlOrRd', alpha=0.6, zorder=2)
            cbar_box = plt.colorbar(cf_box, ax=ax, shrink=0.8, pad=0.02)
            cbar_box.set_label(f'{EMISSION_TYPE} concentration (µg/m³)')

    box_lons = [west, east, east, west, west]
    box_lats = [south, south, north, north, south]
    ax.plot(box_lons, box_lats, color='blue', linewidth=2.5,
            label='Exposure box', zorder=4)

    ax.plot(BNA_LON, BNA_LAT, 'w^', ms=14,
            markeredgecolor='k', markeredgewidth=2,
            zorder=5, label='BNA Airport')

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Exposure Box — {EMISSION_TYPE}")
    ax.legend(loc='upper right', fontsize=9,
              facecolor='white', edgecolor='grey', framealpha=0.9)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    # --- Box summary ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
**Box geometry**
- North: **{north:.4f}**
- South: **{south:.4f}**
- East:  **{east:.4f}**
- West:  **{west:.4f}**
- Area:  **{area_km2:.3f} km²** ({area_m2:,.0f} m²)

**Concentration in box ({EMISSION_TYPE})**
- Mean: **{mean_conc:.4f} µg/m³**
- Max:  **{max_conc:.4f} µg/m³**
- Min:  **{min_conc:.4e} µg/m³**
""")
    with col_b:
        st.markdown(f"""
**Exposure over {exposure_hours:.2f} hr**
- Mean exposure: **{mean_exposure:.4f} µg·hr/m³**
- Max exposure:  **{max_exposure:.4f} µg·hr/m³**

**Population screening estimate**
- Population in box: **{population:,}**
- Population exposure: **{population_exposure:,.2f} person·µg·hr/m³**
""")
    # ------------------------------------------------------------
    # EXPOSURE vs EMISSION PERCENTAGE
    # ------------------------------------------------------------
    st.subheader("5. Exposure sensitivity to emission rate")
    st.caption("How box exposure scales if actual emissions are lower than the "
               "entered rate. Useful for sensitivity analysis or representing "
               "reduced-operations scenarios. 100% = current input.")

    pct = np.linspace(0, 100, 101)          # 0% to 100% in 1% steps
    scale = pct / 100.0
    mean_exp_curve = mean_exposure * scale
    max_exp_curve  = max_exposure  * scale

    fig5, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Mean exposure panel
    ax_a.plot(pct, mean_exp_curve, color='steelblue', lw=2.2)
    ax_a.fill_between(pct, mean_exp_curve, alpha=0.15, color='steelblue')
    ax_a.axvline(100, color='grey', lw=1, ls=':')
    ax_a.scatter([100], [mean_exposure], color='steelblue',
                 s=60, zorder=5, label=f'Current: {mean_exposure:.3f}')
    ax_a.set_xlabel('Emission rate (% of current)')
    ax_a.set_ylabel(f'Mean {EMISSION_TYPE} exposure (µg·hr/m³)')
    ax_a.set_title(f'Mean Exposure vs Emission %\n({exposure_hours:.2f} hr window)')
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(bottom=0)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=9, loc='upper left')

    # Max exposure panel
    ax_b.plot(pct, max_exp_curve, color='crimson', lw=2.2)
    ax_b.fill_between(pct, max_exp_curve, alpha=0.15, color='crimson')
    ax_b.axvline(100, color='grey', lw=1, ls=':')
    ax_b.scatter([100], [max_exposure], color='crimson',
                 s=60, zorder=5, label=f'Current: {max_exposure:.3f}')
    ax_b.set_xlabel('Emission rate (% of current)')
    ax_b.set_ylabel(f'Max {EMISSION_TYPE} exposure (µg·hr/m³)')
    ax_b.set_title(f'Max Exposure vs Emission %\n({exposure_hours:.2f} hr window)')
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(bottom=0)
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(fontsize=9, loc='upper left')

    # Optional NAAQS reference line on max panel (only if current max is within range)
    if NAAQS_VALUE is not None:
        naaqs_exp = NAAQS_VALUE * exposure_hours
        if naaqs_exp <= max_exp_curve.max() * 1.2:
            ax_b.axhline(naaqs_exp, color='darkorange', lw=1.3, ls='--',
                         label=f'NAAQS ({NAAQS_VALUE} µg/m³ × {exposure_hours:.1f} hr)')
            ax_b.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

# ------------------------------------------------------------
# SUMMARY TABLE
# ------------------------------------------------------------
st.subheader("Summary")

summary_rows = []
for d in [0.5, 1, 2, 5, 10, 15, 20]:
    c = gaussian_plume(d*1000, 0, Q_g_s, U_WIND, STABILITY) * 1e6
    summary_rows.append({"Distance (km)": d, "Centreline (µg/m³)": fmt_conc(c)})

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
**Inputs**
- Emission type: **{EMISSION_TYPE}**
- Emission rate: **{Q_g_s:.4f} g/s** ({Q_tons_day:.4f} MT/day)
- Wind speed: **{U_WIND} m/s**
- Stability class: **{STABILITY}**
- Wind from: **{WIND_DIR_DEG}°** → plume to **{plume_dir}**
""")
    if NAAQS_VALUE is not None:
        st.markdown(f"- NAAQS reference: **{NAAQS_VALUE} µg/m³**")
with col2:
    st.markdown("**Centreline concentration vs distance**")
    st.table(summary_rows)

st.caption("Screening-level Gaussian plume model. Not for regulatory use. "
           "Pasquill-Gifford dispersion coefficients; ground-level point source assumed.")
