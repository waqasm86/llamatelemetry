"""
llamatelemetry.kaggle.graphistry - Auto-register pygraphistry from Kaggle secrets.

Extracted from notebook 6-8 patterns to absorb boilerplate.
"""


def auto_register_graphistry() -> bool:
    """
    Load Graphistry credentials from Kaggle secrets and call ``graphistry.register()``.

    Tries these credential sets in order:
        1. ``Graphistry_Personal_Key_ID`` + ``Graphistry_Personal_Secret_Key``
        2. ``GRAPHISTRY_USERNAME`` + ``GRAPHISTRY_PASSWORD``
        3. ``GRAPHISTRY_API_KEY``

    Returns:
        True if pygraphistry was registered successfully.
    """
    from .secrets import KaggleSecrets

    secrets = KaggleSecrets()

    key_id = secrets.get("Graphistry_Personal_Key_ID")
    key_secret = secrets.get("Graphistry_Personal_Secret_Key")

    if key_id and key_secret:
        try:
            import graphistry

            graphistry.register(
                api=3,
                protocol="https",
                server="hub.graphistry.com",
                personal_key_id=key_id,
                personal_key_secret=key_secret,
            )
            return True
        except Exception:
            return False

    username = secrets.get("GRAPHISTRY_USERNAME")
    password = secrets.get("GRAPHISTRY_PASSWORD")
    if username and password:
        try:
            import graphistry

            graphistry.register(
                api=3,
                protocol="https",
                server="hub.graphistry.com",
                username=username,
                password=password,
            )
            return True
        except Exception:
            return False

    api_key = secrets.get("GRAPHISTRY_API_KEY")
    if api_key:
        try:
            import graphistry

            graphistry.register(api=3, token=api_key)
            return True
        except Exception:
            return False

    return False
