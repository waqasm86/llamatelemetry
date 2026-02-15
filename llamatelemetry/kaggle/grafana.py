"""
llamatelemetry.kaggle.grafana - Auto-configure Grafana Cloud from Kaggle secrets.

Extracted from notebook 14-16 patterns to absorb boilerplate.
"""

from typing import Optional


def auto_configure_grafana_cloud() -> bool:
    """
    Load OTLP endpoint + token from Kaggle secrets and call ``configure()``.

    Expects the following Kaggle secrets:
        - ``OTLP_ENDPOINT``: Grafana Cloud OTLP endpoint URL.
        - ``OTLP_TOKEN``: Grafana Cloud API token (used as Authorization header).

    Returns:
        True if Grafana Cloud was configured successfully.
    """
    from .secrets import KaggleSecrets

    secrets = KaggleSecrets()
    endpoint = secrets.get("OTLP_ENDPOINT")
    token = secrets.get("OTLP_TOKEN")

    if not endpoint:
        return False

    headers = {}
    if token:
        headers["Authorization"] = f"Basic {token}"

    try:
        import llamatelemetry

        llamatelemetry.configure(
            otlp_endpoint=endpoint,
            otlp_headers=headers,
        )
        return True
    except Exception:
        return False
