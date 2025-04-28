from collections import defaultdict

# Recursive defaultdict constructor
def nested_dict():
    return defaultdict(nested_dict)

# Initialize the structure
allRoiPix = nested_dict()

# Dummy roiPix data (can be anything – here it's just a list)
roiPix = ["pixel1", "pixel2", "pixel3"]

# Assign to nested indices
iCh = 0
i_scanpath = 0
i_roi = 0
iDepth = 0

# Perform the assignment
allRoiPix[iCh][i_scanpath][i_roi][iDepth] = roiPix

# Add another example to show uneven growth
allRoiPix[0][1][0][0] = ["another_roiPix"]
allRoiPix[1][0][0][0] = ["yet_another_roiPix"]

# Type checking at each level
print("Top level:", type(allRoiPix))                      # defaultdict
print("Level 1 (allRoiPix[0]):", type(allRoiPix[0]))      # defaultdict
print("Level 2 (allRoiPix[0][0]):", type(allRoiPix[0][0]))  # defaultdict
print("Level 3 (allRoiPix[0][0][0]):", type(allRoiPix[0][0][0]))  # defaultdict
print("Value (allRoiPix[0][0][0][0]):", allRoiPix[0][0][0][0])

# Print entire structure
import pprint
print("\nFull structure:")
pprint.pprint(allRoiPix)  
