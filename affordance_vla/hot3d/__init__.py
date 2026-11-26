"""HOT3D-Clips (Meta's egocentric hand-object dataset) read as a source of interaction labels: AFFO-11.

clips.py reads one clip, contact.py finds where a hand touches an object and how the fingers oppose,
render.py draws the results. Needs the hot3d environment (envs/server/setup_hot3d_env.sh); the rest of the package
does not import this one.
"""
