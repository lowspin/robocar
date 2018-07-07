import pygame
import numpy as np
import scipy, scipy.ndimage
import time
import json
import glob




image_paths = glob.glob('./frames/*.png')
image_paths.sort()
print('Found %d frames to label.' % len(image_paths))

# Check resolution.
TRUE_X = 128
TRUE_Y = 96
test_image = scipy.ndimage.imread(image_paths[0])
assert test_image.shape[0] == TRUE_Y, test_image.shape
assert test_image.shape[1] >= TRUE_X, test_image.shape

RES_SCALE = 8
RES_X = int(TRUE_X * RES_SCALE)
RES_Y = int(TRUE_Y * RES_SCALE)
assert float(TRUE_X) / TRUE_Y == float(RES_X) / RES_Y

pygame.init()
display = pygame.display.set_mode((RES_X, RES_Y))


class ImageGrabber:

	def __init__(self):
		self.i = -1

	def __call__(self):
		self.i += 1
		if self.i >= len(image_paths):
			self.i = 0
		image_path = image_paths[self.i]
		image = scipy.ndimage.imread(image_path)[:, :128, :]
		return scipy.misc.imresize(image, (RES_Y, RES_X)), image_path

get_image = ImageGrabber()

def draw_image(scaled):
	return pygame.transform.rotate(pygame.surfarray.make_surface(
		np.flipud(scaled)
	), 90)


# Set indices, flags, and accumulators.
new_image = True
running = True
class_number = None
diameter = 50
data = {}


# Print instructions:
datafile_name = 'data_%s.json' % time.time()
print '''
Instructions:
=============
Key/Click      Action
Click          Set box center.
Right-click    Set box radius using top edge.
Number keys    Assign box class.
Tab            Save and continue.
               Will only continue if no labeling is set
               or both location and class are set.
Space          Clear labeling.
ESC/Window X   Exit and save data to %s.
''' % datafile_name


start_time = time.time()
while running:

	events = pygame.event.get()

	for event in events:

		if event.type == pygame.MOUSEBUTTONUP:

			# Choose box diameter using top edge.
			if event.button == 3:
				top_point = pygame.mouse.get_pos()
				if center_point is not None:
					diameter = 2 * abs(top_point[1] - center_point[1])
			
			# Choose center point.
			else:
				center_point = pygame.mouse.get_pos()
				
		elif event.type == pygame.KEYUP:
			key = event.key
			try:

				# Set class number.
				if chr(key) in '0123456789':
					class_number = int(chr(key))

				# Clear the markings.
				elif chr(key) == ' ':
					center_point = None
					class_number = None

				# Save.
				elif key == pygame.K_TAB:
					# Only save if both centerpoint and label are none or set.
					cp = center_point is None
					cl = class_number is None
					if (cp and cl) or (not cp and not cl):

						if not cp:
							center_point = tuple(float(z) / RES_SCALE for z in center_point)

						# Save the labeling.
						data[image_path] = (
							center_point, 
							diameter / RES_SCALE if center_point is not None else None, 
							class_number, 
						)

						# Continue
						new_image = True
						center_point = None
						#class_number = None

				elif key == pygame.K_ESCAPE:
					running = False

			except ValueError:
				pass

		elif event.type == pygame.QUIT:
			running = False

	# Draw
	if new_image:
		scaled, image_path = get_image()
		new_image = False
		center_point = None
	image_surface = draw_image(scaled)
	display.blit(image_surface, (0, 0))

	if center_point is not None:
		xc, yc = center_point
		x = xc - diameter / 2.
		y = yc - diameter / 2.
		x2 = xc + diameter / 2.
		y2 = yc + diameter / 2.
		dx = x2 - x
		dy = y2 - y
		pygame.draw.rect(image_surface, (0, 0, 128), [x, y, dx, dy], 4)
		display.blit(image_surface, (0, 0))
		corners = [None, None]

	if class_number is not None:
		font = pygame.font.SysFont('Sans Bold', 30)
		text_surface = font.render('#%d' % (class_number,), False, (255, 255, 255))
		display.blit(text_surface, (10, 10))

	pygame.display.update()

end_time = time.time()

# Save data.
print 'Got %d positive labelings from %d images in %s seconds.' % (
	len([v for v in data.values() if v[0] is not None]),
	get_image.i, end_time - start_time
)
j = json.dumps(data)
with open(datafile_name, 'w') as f:
	f.write(j)
print 'Saved to %s.' % datafile_name

pygame.quit()
