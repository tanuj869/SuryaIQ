"""India choropleth helper — maps the master panel onto the GeoJSON boundaries."""
import json
from pathlib import Path

import plotly.graph_objects as go

GEOJSON = Path(__file__).parent / "assets" / "india_states.geojson"
PRIMARY = "#F2A900"
SEQ = "Oranges"

# canonical state -> GeoJSON NAME_1
_TO_GEO = {
    "Odisha": "Orissa", "Uttarakhand": "Uttaranchal",
    "Andaman & Nicobar": "Andaman and Nicobar",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "Dadra & Nagar Haveli and Daman & Diu": "Dadra and Nagar Haveli",  # best fit
    "Ladakh": "Jammu and Kashmir",  # pre-2019 geom includes Ladakh
    "Telangana": "Andhra Pradesh",  # pre-2014 geom
}


def to_geo(name):
    return _TO_GEO.get(name, name)


_cache = None
def geojson():
    global _cache
    if _cache is None:
        _cache = json.loads(GEOJSON.read_text())
    return _cache


def choropleth(df, metric, title, colorbar_title=None, hover_extra=None):
    """Build a Plotly choropleth of `metric` colored across Indian states.
    df must have 'state' and metric columns."""
    d = df.copy()
    d["geo_name"] = d["state"].map(to_geo)
    hover = ["state", metric] + (hover_extra or [])
    custom = d[hover].values
    fig = go.Figure(go.Choropleth(
        geojson=geojson(),
        locations=d["geo_name"],
        z=d[metric],
        featureidkey="properties.NAME_1",
        colorscale=SEQ,
        marker_line_color="#1F1F1F", marker_line_width=0.5,
        colorbar=dict(title=colorbar_title or metric, thickness=12, len=0.7),
        hovertemplate=("<b>%{customdata[0]}</b><br>" +
                       f"{metric}: " + "%{z:.3f}<extra></extra>"),
        customdata=custom,
    ))
    # Explicit lat/lon range for India (prevents the plotly re-render shrink bug
    # that fitbounds="locations" triggers on tab-switch / scroll)
    fig.update_geos(
        visible=False, bgcolor="rgba(0,0,0,0)",
        projection_type="mercator",
        lataxis_range=[6, 37], lonaxis_range=[68, 98],
    )
    fig.update_layout(title=title, title_font_size=16,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="white", margin=dict(l=0, r=0, t=40, b=0),
                      height=560, width=1000, autosize=False)
    return fig
