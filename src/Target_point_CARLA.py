from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref
import matplotlib.pyplot as plt
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla

from carla import ColorConverter as cc
render_lanes = 0
import logging
import random
import re
import weakref
import numpy as np
import cv2
import time



def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image


im_id = 1
tt = 0
# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================
#          F1      restart
#          F5      toggle _autopilot_enabled
#          F9      recording
#          F11     next_weather    reversed
#          F12     next_weather
#          0-9     set_sensors
#          w           throtle
#    a         d       steer 
#       z              reverse control
#         
#          s       brake
#          h       hand-brake
#          tab     toggle camera
class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        world.vehicle.set_autopilot(self._autopilot_enabled)
        #world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, world, keys):
        if(True):
                if (keys == 65470):   # F1, event.key == K_BACKSPACE:
                    world.restart()
                elif (keys == 9):   # TAB, event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif (keys == 65480):   # F11, event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif (keys == 65481):   # F12, event.key == K_c:
                    world.next_weather()
                elif (keys == ord('0')):   # event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif (keys == ord('1')):   # event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(1 - 1)
                elif (keys == ord('2')):   # event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(2 - 1)
                elif (keys == ord('3')):   # event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(3 - 1)
                elif (keys == ord('4')):   # event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(4 - 1)
                elif (keys == ord('5')):   # event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(5 - 1)
                elif (keys == ord('6')):   # event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(6 - 1)
                elif (keys == ord('7')):   # event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(7 - 1)
                elif (keys == ord('8')):   # event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(8 - 1)
                elif (keys == ord('9')):   # event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(9 - 1)
                elif (keys == 65478):   # F9, event.key == K_r:
                    world.camera_manager.toggle_recording()
                elif (keys == ord('z')):   # Z, event.key == K_q:
                    self._control.reverse = not self._control.reverse
                elif (keys == 65474):   # F5, event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.vehicle.set_autopilot(self._autopilot_enabled)
                    world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            self._parse_keys( key, int(round(time.time() * 1000)) )  # pygame.key.get_pressed(), clock.get_time())
            world.vehicle.apply_control(self._control)

    def _parse_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys == ord('w') else 0.0
        steer_increment = 5e-4 * milliseconds
        if (keys == ord('a')):   # keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif (keys == ord('d')):   # keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys == ord('s') else 0.0
        self._control.hand_brake = True if keys == ord('h') else False  # keys[K_SPACE]

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height, name = "raw"):
        self.dim = (width, height)
        self._text = 'Autopilot off'
        self.render_lanes = 0
        self.name =  name
        self.left_a, self.left_b, self.left_c = [],[],[]
        self.right_a, self.right_b, self.right_c = [],[],[]
        self._info_text = []


    def Lanes_detection(self, img, img_org, nwindows=50, margin=20, minipix = 5, draw_windows=False):
       
        # ==============================================================================
        # -- Implement the Lanes_detection function here -------------------------------
        # ==============================================================================

        return img_2, img_2_o

    

    def tick(self, world, clock):
        pass

    def notification(self, text, seconds=2.0):
        self._text = text

    def error(self, text):
        self._text = text


    def get_curve(self, img, leftx, rightx):
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        y_eval = np.max(ploty)
        ym_per_pix = 30.5/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/720 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        car_pos = img.shape[1]/2
        l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center = (car_pos - lane_center_position) * xm_per_pix / 10
        # Now our radius of curvature is in meters
        return (left_curverad, right_curverad, center)

    def wrap_image(self, img, inverse=False):

        img_size = (img.shape[1], img.shape[0])

        bot_width = .96
        mid_width = .2
        height_pct = .56
        bottom_trim = .935
        offset = img_size[0] * .15
        if inverse:
            dst = np.float32([[img.shape[1]*(.5-mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+bot_width/2), img.shape[0]*bottom_trim], [img.shape[1]*(.5-bot_width/2), img.shape[0]*bottom_trim]])
            src = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
        else:
            src = np.float32([[img.shape[1]*(.5-mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+bot_width/2), img.shape[0]*bottom_trim], [img.shape[1]*(.5-bot_width/2), img.shape[0]*bottom_trim]])
            dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])

        M = cv2.getPerspectiveTransform(src, dst)
        wraped_img = cv2.warpPerspective(img, M, (img_size[0], img_size[1]))

        return wraped_img

    def draw_lanes(self, img, left_fit, right_fit, middle_fit):

        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        color_img = np.zeros_like(img)
        
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        mid_1 = np.array([np.transpose(np.vstack([middle_fit, ploty]))])
        mid_2 = np.array([np.flipud(np.transpose(np.vstack([middle_fit, ploty])))])

        points_1 = np.hstack((left, mid_2))
        points_r = np.hstack((mid_1, right))
        
        cv2.fillPoly(color_img, np.int_(points_1), (0,200,0))
        cv2.fillPoly(color_img, np.int_(points_r), (0,0,255))

        cv2.drawContours(color_img, [mid_1.reshape((img.shape[0],2)).astype(int)], 0, (255, 0,0), 2)
        newwarp = self.wrap_image(color_img, inverse=True)

        inv_perspective = cv2.addWeighted(img, 1, newwarp, 0.7, 0)

        mask = (newwarp[:,:,0] == 255).astype(float) * 255

        nonzero = mask.nonzero()

        radius = 2
        color = (0, 255, 0)

        thickness = 20
        center_coordinates = (nonzero[1][0], nonzero[0][0])
        center_coordinates_t = (nonzero[1][0] - 20, nonzero[0][0] - 20)

        inv_perspective = cv2.circle(inv_perspective, center_coordinates, radius, color, thickness)

        fontScale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2

        inv_perspective = cv2.putText(inv_perspective, 'Target point', center_coordinates_t, font, fontScale, (0, 255, 255), thickness, cv2.LINE_AA)
        
        return inv_perspective, newwarp


    def get_hist(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = np.sum(gray[int(img.shape[0]/2):, :], axis=0)
        return hist


    def render(self, semantic_image, raw_image):
        if(raw_image is not None):

            mask = (semantic_image[:, :, 1] == 234).astype(float) * 255
            mask_b = (semantic_image[:, :, 1] == 234).astype(np.uint8) * 255

            mask = mask[:, :, np.newaxis]
            mask_b = mask_b[:, :, np.newaxis]

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            dilation = cv2.dilate(mask_b, kernel, iterations = 1)
            dilation = dilation[:, :, np.newaxis]   
            dilation = np.repeat(dilation, 3, axis=2)
 

            mask2 = np.zeros_like(raw_image)
            height, width, _ = raw_image.shape
 
            bot_width = .96
            mid_width = .2
            height_pct = .54
            bottom_trim = .935
            hexagon = np.array([
                                 [(int(raw_image.shape[1]*(.5-mid_width/2)), int(raw_image.shape[0]*height_pct)),
                                 (int(raw_image.shape[1]*(.5+mid_width/2)), int(raw_image.shape[0]*height_pct)),
                                 (int(raw_image.shape[1]*(.5+bot_width/2)), int(raw_image.shape[0]*bottom_trim)),
                                 (int(raw_image.shape[1]*(.5-bot_width/2)), int(raw_image.shape[0]*bottom_trim))]
                              ])
            print(hexagon)

            mask2 = cv2.fillPoly(mask2, hexagon, 255)
            mask = cv2.bitwise_and(dilation.astype(float), mask2.astype(float))

            try:
                out_img, raw_image = self.Lanes_detection(mask, raw_image)
          
            except:
                pass

            cv2.imshow('Segmented Lanes {}'.format(self.name), raw_image)
            #cv2.imshow('Triangle Mask {}'.format(self.name), depth_image)
            cv2.imshow('Triangle Mask {}'.format(self.name), mask)


        key = cv2.waitKeyEx(30)
        if(key == 27):
            cv2.destroyAllWindows()
        return (key), mask


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = ' '.join(event.other_actor.type_id.replace('_', '.').title().split('.')[1:])
        if(self._hud is not None):
            self._hud.notification('Collision with %r' % actor_type)
        else:
            print('Collision with %r' % actor_type)

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


START_POSITION = carla.Transform(carla.Location(x=180.0, y=199.0, z=40.0))


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.hud = hud
        blueprint = self._get_random_blueprint()
        self.vehicle = self.world.spawn_actor(blueprint, START_POSITION)
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        self.camera_manager = CameraManager(self.vehicle, self.hud, args)
        self.camera_manager.set_sensor(0, notify=False)
        self.camera_manager_semantic = CameraManager(self.vehicle, self.hud, args)
        self.camera_manager_semantic.set_sensor(5, notify=False)
        self.controller = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.args = args

    def restart(self):
        cam_index = self.camera_manager._index
        cam_pos_index = self.camera_manager._transform_index


        start_pose = self.vehicle.get_transform()
        start_pose.location.z += 2.0
        start_pose.rotation.roll = 0.0
        start_pose.rotation.pitch = 0.0
        blueprint = self._get_random_blueprint()
        self.destroy()
        self.vehicle = self.world.spawn_actor(blueprint, start_pose)
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        self.camera_manager = CameraManager(self.vehicle, self.hud, self.args)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager_semantic._transform_index = cam_pos_index

        
        actor_type = ' '.join(self.vehicle.type_id.replace('_', '.').title().split('.')[1:])
        if(self.hud is not None):
            self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        if(self.hud is not None):
            self.hud.notification('Weather: %s' % preset[1])
        self.vehicle.get_world().set_weather(preset[0])

    def tick(self, clock):
        if(self.hud is not None):
            self.hud.tick(self, clock)

    def render(self, display):
        ret = 0
        image = self.camera_manager.render(display)
        image_semantic  = self.camera_manager_semantic.render(display)

        #image_depth = self.camera_manager_depth.render(display)

        if(self.hud is not None):
            self.hud.name = "raw"
            ret, segm = self.hud.render(image_semantic, image)


        return ret, image, segm

    def destroy(self):
        for actor in [self.camera_manager.sensor, self.collision_sensor.sensor, self.vehicle]:
            if actor is not None:
                actor.destroy()

    def _get_random_blueprint(self):
        bp = random.choice(self.world.get_blueprint_library().filter('vehicle').filter('model3'))
        # jeff, use jeep only
        print("bp", bp.tags)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, args):
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))]
        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if(hud is not None):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            else:  # 1280x720
                bp.set_attribute('image_size_x', str(args.width) )
                bp.set_attribute('image_size_y', str(args.height) )
            item.append(bp)
        self._index = None
        #self._server_clock = pygame.time.Clock()
        self._image_raw = None
        #print('_server_clock', self._server_clock)

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify and (self._hud is not None):
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        if(self._hud is not None):
            self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        #if self._surface is not None:
        #    display.blit(self._surface, (0, 0))
        return (self._image_raw)

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        
        image.convert(self._sensors[self._index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        #print(self._index)
        array = array[:, :, :3]
        
        self._image_raw = array.copy()

        return
        '''
        array = array[:, :, ::-1]
        self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self._recording:
            image.save_to_disk('_out/%08d' % image.frame_number)
        '''


# ==============================================================================
# -- mc_args -------------------------------------------------------------
# ==============================================================================

class mc_args(object):
    def __init__(self):
        self.debug = True
        self.host = '127.0.0.1'
        self.port = 2000
        self.autopilot = True  # True
        self.width =  460  # 1280
        self.height = 360   # 720
args = mc_args()

def make_points(image, average): 
    slope, y_int = average 
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


client = carla.Client(args.host, args.port)
client.set_timeout(2.0)
hud = HUD(args.width, args.height)


# 200.
world = None
try:
    world = World(client.get_world(), hud, args)   
    world.camera_manager.toggle_camera()
    world.camera_manager_semantic.toggle_camera()
    controller = KeyboardControl(world, args.autopilot)
    tt = 0
    im_id = 0
    while ( True ):
        key, img, seg= world.render(None)
        #print(img.shape)
        if tt >=100:
            im_id += 1
            cv2.imwrite("images/img_{}.png".format(im_id), seg)     
            tt = 0
            #print(im_id)
        tt+=1
        controller.parse_events(world, key)
        if (key == 27):
            break
    
finally:  
    if world is not None:
        world.destroy()