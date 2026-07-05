#uv run generate_image_db.py -i data/naodevils_data/training -l True -o naodevils_training
uv run generate_image_db.py -i data/naodevils_data/validation -l True -o naodevils_validation

uv run generate_image_db.py -i data/go26_patches -l True -o go26_patches