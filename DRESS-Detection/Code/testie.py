import openslide

# Load the WSIfile
slide_path = "Dataset\OSU\MDE\S12-35327  G-3.svs"


slide = openslide.OpenSlide(slide_path)

# Get objective magnification (base magnification)
objective_power = slide.properties.get("openslide.objective-power")

# Convert to integer if not None
if objective_power is not None:
    objective_power = int(objective_power)  # Convert string to int
else:
    raise ValueError("Objective power not found in metadata.")

# Get available levels and downsample factors
levels = slide.level_dimensions
downsamples = slide.level_downsamples

# Calculate approximate magnifications per level
level_magnifications = [int(objective_power / float(ds)) for ds in downsamples]

print(f"Objective Magnification: {objective_power}x")
print(f"Available Levels: {levels}")
print(f"Downsample Factors: {downsamples}")
print(f"Available Magnifications: {level_magnifications}")
