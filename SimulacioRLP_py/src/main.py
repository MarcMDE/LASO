from __future__ import division

#from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
from panda3d.core import Texture
from panda3d.core import KeyboardButton
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerQueue, CollisionRay
from panda3d.core import Material, LRotationf, NodePath
from panda3d.core import AmbientLight, DirectionalLight, Spotlight
from panda3d.core import TextNode
from panda3d.core import LVector3, BitMask32
from panda3d.core import *
from direct.gui.OnscreenText import OnscreenText
from direct.interval.MetaInterval import Sequence, Parallel
from direct.interval.LerpInterval import LerpFunc
from direct.interval.FunctionInterval import Func, Wait

from direct.task.Task import Task
import sys
from recognition import *

from pid import *
from aStar import *
import math

import time

from panda3d.core import *
import sys
import os

from direct.showbase.ShowBase import ShowBase
from direct.interval.IntervalGlobal import *
from direct.gui.DirectGui import OnscreenText
from direct.showbase.DirectObject import DirectObject
from direct.actor import Actor
from random import *

from constants import *

import numpy as np
import threading as th
import speechRecognition as sr
import _thread
import threading
import cv2

#GOOGLE CLOUD SPEECH

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "robotica-272015-91d7b22ba751.json"


import re
import sys


#from google.cloud import storage

from google.cloud import storage
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

vr = sr.VoiceRecognition(0)

action = "start"
voice_solving = False
dir_veu = (0,0)
th = None


# Some constants for the program
ACCEL = 70         # Acceleration in ft/sec/sec
MAX_SPEED = 5      # Max speed in ft/sec
MAX_SPEED_SQ = MAX_SPEED ** 2  # Squared to make it easier to use lengthSquared
# Instead of length

timerStatus = 'stop'


def finalitzar():
    global th
    if th is not None:
        th.join()
    print()
    sys.exit()

def listen_print_loop(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    global action
    global voice_solving
    global dir_veu
    dir = (0, 0)

    num_chars_printed = 0

    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]

        if not result.alternatives:
            continue
        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        tranSlipt = transcript.split()

        if 'empezar' in tranSlipt or 'comenzar' in tranSlipt:
            print("Game is runing")
            action = "start"

        elif 'para' in tranSlipt or 'parar' in tranSlipt:
            print("Game finished")
            action = "stop"

        elif 'reiniciar' in tranSlipt or 'reinicia' in tranSlipt:
            action = "restart"

        elif 'comandos' in tranSlipt or 'comando' in tranSlipt:
            print("VOICE SOLVING ACTIVATED")
            voice_solving = True
            dir = (0, 0)

        elif 'automático' in tranSlipt or 'automatico' in tranSlipt:
            print("VOICE SOLVING DEACTIVATED")
            voice_solving = False


        elif voice_solving:
            p = dir[0]
            r = dir[1]
            if 'arriba' in tranSlipt:
                p = -1
            if 'abajo' in tranSlipt:
                p = 1
            if 'izquierda' in tranSlipt:
                r = -1
            if 'derecha' in tranSlipt:
                r = 1
            if 'recto' in tranSlipt:
                p = 0
                r = 0

            dir = (p, r)
            dir_veu = dir
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:

            print(transcript + overwrite_chars)


             #Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break

            num_chars_printed = 0

def listenVoice():

    language_code = 'es-ES'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    with sr.MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        listen_print_loop(responses)


class BallInMazeDemo(ShowBase):
    def __init__(self):
        global action
        # Initialize the ShowBase class from which we inherit, which will
        # create a window and set up everything we need for rendering into it.
        action="start"
        ShowBase.__init__(self)
        winProps = WindowProperties()
        winProps.setTitle("LASO Simulation - (RLP 2019-2020)")
        base.win.requestProperties(winProps)

        if not base.win.getGsg().getSupportsBasicShaders():
            print("Error: Video driver reports that depth textures are not supported.")
            return
        if not base.win.getGsg().getSupportsDepthTexture():
            print("Error: Video driver reports that depth textures are not supported.")
            return

        self.accept("escape", finalitzar)  # Escape quits

        # Disable default mouse-based camera control.  This is a method on the
        # ShowBase class from which we inherit.
        #self.disableMouse()

        # Place the camera
        camera.setPosHpr(0, 0, 80, 0, -90, 0)

        base.trackball.node().set_pos(0, 200, 0)
        base.trackball.node().set_hpr(0, 60, 0)

        #camera.setPosHpr(0, 0, 25, 0, -90, 0)
        #base.camLens.setNearFar(200, 600)

        #self.enableMouse()

        # Load the maze and place it in the scene
        #self.maze = loader.loadModel("models/maze")

        self.skybox = loader.loadModel("models/skybox")
        self.skybox.reparentTo(render)
        self.skybox.setPosHpr(0, 0, 45+ MAZE_OFFSETS[CURR_MAZE], 0, -180, 0)
        #self.skybox.setScale(2.54)

        self.taula = loader.loadModel("models/taula")
        self.taula.reparentTo(render)
        self.taula.setPosHpr(0, 0, -53.6 + MAZE_OFFSETS[CURR_MAZE], 0, 0, 0)
        self.taula.setScale(1.8)


        self.laso_box = loader.loadModel("models/laso_box")
        self.laso_box.reparentTo(render)
        self.laso_box.setPosHpr(0, 0, MAZE_OFFSETS[CURR_MAZE], 0, 0, 0)

        self.laso_ax = loader.loadModel("models/laso_ax")
        self.laso_ax.reparentTo(self.laso_box)
        self.laso_ax.setPosHpr(0, 0, MAZE_HEIGHT, 0, 0, 0)

        self.maze_i = CURR_MAZE

        self.maze = loader.loadModel("models/" + MAZES_NAME[self.maze_i])
        self.maze.reparentTo(render)
        self.maze.setPosHpr(0, 0, MAZE_HEIGHT, 0, 0, 0)
        #self.maze.setScale(2.54)

        # Load custom maze

        #self.maze2 = loader.loadModel("models/lab4")
        #self.maze2.reparentTo(render)


        # Most times, you want collisions to be tested against invisible geometry
        # rather than every polygon. This is because testing against every polygon
        # in the scene is usually too slow. You can have simplified or approximate
        # geometry for the solids and still get good results.
        #
        # Sometimes you'll want to create and position your own collision solids in
        # code, but it's often easier to have them built automatically. This can be
        # done by adding special tags into an egg file. Check maze.egg and ball.egg
        # and look for lines starting with <Collide>. The part is brackets tells
        # Panda exactly what to do. Polyset means to use the polygons in that group
        # as solids, while Sphere tells panda to make a collision sphere around them
        # Keep means to keep the polygons in the group as visable geometry (good
        # for the ball, not for the triggers), and descend means to make sure that
        # the settings are applied to any subgroups.
        #
        # Once we have the collision tags in the models, we can get to them using
        # NodePath's find command

        # Find the collision node named wall_collide
        #self.walls = self.maze.find("**/wall_collide")
        self.walls = self.maze.find("**/wall_col")

        # Collision objects are sorted using BitMasks. BitMasks are ordinary numbers
        # with extra methods for working with them as binary bits. Every collision
        # solid has both a from mask and an into mask. Before Panda tests two
        # objects, it checks to make sure that the from and into collision masks
        # have at least one bit in common. That way things that shouldn't interact
        # won't. Normal model nodes have collision masks as well. By default they
        # are set to bit 20. If you want to collide against actual visable polygons,
        # set a from collide mask to include bit 20
        #
        # For this example, we will make everything we want the ball to collide with
        # include bit 0
        self.walls.node().setIntoCollideMask(BitMask32.bit(0))
        # CollisionNodes are usually invisible but can be shown. Uncomment the next
        # line to see the collision walls
        #self.walls.show() # Show wall colliders

        # We will now find the triggers for the holes and set their masks to 0 as
        # well. We also set their names to make them easier to identify during
        # collisions

        self.loseTriggers = []
        #for i in range(6):
        for i in range(3):
            #trigger = self.maze.find("**/hole_collide" + str(i))
            trigger = self.maze.find("**/hole"+str(i+1)+"_col")
            trigger.node().setIntoCollideMask(BitMask32.bit(0))
            trigger.node().setName("loseTrigger")
            self.loseTriggers.append(trigger)
            # Uncomment this line to see the triggers
            #trigger.show() # Show lose triggers colliders

        # Ground_collide is a single polygon on the same plane as the ground in the
        # maze. We will use a ray to collide with it so that we will know exactly
        # what height to put the ball at every frame. Since this is not something
        # that we want the ball itself to collide with, it has a different
        # bitmask.
        #self.mazeGround = self.maze.find("**/ground_collide")
        self.mazeGround = self.maze.find("**/ground_col")
        self.mazeGround.node().setIntoCollideMask(BitMask32.bit(1))

        # Load the ball and attach it to the scene
        # It is on a root dummy node so that we can rotate the ball itself without
        # rotating the ray that will be attached to it
        self.ballRoot = render.attachNewNode("ballRoot")
        self.ball = loader.loadModel("models/bball")
        self.ball.reparentTo(self.ballRoot)

        # Find the collison sphere for the ball which was created in the egg file
        # Notice that it has a from collision mask of bit 0, and an into collison
        # mask of no bits. This means that the ball can only cause collisions, not
        # be collided into
        self.ballSphere = self.ball.find("**/ball")
        self.ballSphere.node().setFromCollideMask(BitMask32.bit(0))
        self.ballSphere.node().setIntoCollideMask(BitMask32.allOff())

        # No we create a ray to start above the ball and cast down. This is to
        # Determine the height the ball should be at and the angle the floor is
        # tilting. We could have used the sphere around the ball itself, but it
        # would not be as reliable
        self.ballGroundRay = CollisionRay()     # Create the ray
        self.ballGroundRay.setOrigin(0, 0, 10)    # Set its origin
        self.ballGroundRay.setDirection(0, 0, -1)  # And its direction
        # Collision solids go in CollisionNode
        # Create and name the node
        self.ballGroundCol = CollisionNode('groundRay')
        self.ballGroundCol.addSolid(self.ballGroundRay)  # Add the ray
        self.ballGroundCol.setFromCollideMask(
            BitMask32.bit(1))  # Set its bitmasks
        self.ballGroundCol.setIntoCollideMask(BitMask32.allOff())
        # Attach the node to the ballRoot so that the ray is relative to the ball
        # (it will always be 10 feet over the ball and point down)
        self.ballGroundColNp = self.ballRoot.attachNewNode(self.ballGroundCol)
        # Uncomment this line to see the ray
        #self.ballGroundColNp.show()  # Show ball collision ray

        # Finally, we create a CollisionTraverser. CollisionTraversers are what
        # do the job of walking the scene graph and calculating collisions.
        # For a traverser to actually do collisions, you need to call
        # traverser.traverse() on a part of the scene. Fortunately, ShowBase
        # has a task that does this for the entire scene once a frame.  By
        # assigning it to self.cTrav, we designate that this is the one that
        # it should call traverse() on each frame.
        self.cTrav = CollisionTraverser()

        # Collision traversers tell collision handlers about collisions, and then
        # the handler decides what to do with the information. We are using a
        # CollisionHandlerQueue, which simply creates a list of all of the
        # collisions in a given pass. There are more sophisticated handlers like
        # one that sends events and another that tries to keep collided objects
        # apart, but the results are often better with a simple queue
        self.cHandler = CollisionHandlerQueue()
        # Now we add the collision nodes that can create a collision to the
        # traverser. The traverser will compare these to all others nodes in the
        # scene. There is a limit of 32 CollisionNodes per traverser
        # We add the collider, and the handler to use as a pair
        self.cTrav.addCollider(self.ballSphere, self.cHandler)
        self.cTrav.addCollider(self.ballGroundColNp, self.cHandler)

        # Collision traversers have a built in tool to help visualize collisions.
        # Uncomment the next line to see it.
        #self.cTrav.showCollisions(render)  # Show traveser collisions


        # This section deals with lighting for the ball. Only the ball was lit
        # because the maze has static lighting pregenerated by the modeler
        self.alightColor = 0.1
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((self.alightColor, self.alightColor, self.alightColor, 1))
        #ambientLight.setColor((1, 0, 0, 1))
        self.ambientL = render.attachNewNode(ambientLight)
        render.setLight(self.ambientL)
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(LVector3(0, 1, 0))
        #directionalLight.setColor((0.375, 0.375, 0.375, 1))
        directionalLight.setColor((1, 1, 1, 1))
        directionalLight.setSpecularColor((1, 1, 1, 1))
        #self.directL = render.attachNewNode(directionalLight)
        #render.setLight(self.directL)

        """
        self.teapot = loader.loadModel('teapot')
        self.teapot.reparentTo(render)
        self.teapot.setPos(0, 0, 0)
        self.teapotMovement = self.teapot.hprInterval(50, LPoint3(0, 360, 360))
        self.teapotMovement.loop()
        """

        self.lightColor = 1000
        self.light = render.attachNewNode(Spotlight("Spot"))
        self.light.node().setScene(render)
        self.light.node().setShadowCaster(True, 1024, 1024)
        self.light.node().setAttenuation((1, 0, 1))
        #self.light.node().showFrustum()
        self.light.node().getLens().setFov(48)
        self.light.node().getLens().setNearFar(5, 500)
        #self.light.node().setColor((100000, 100000, 100000, 1))
        self.light.node().setColor((self.lightColor, self.lightColor, self.lightColor, 1))
        self.light.setPos(0, 0, 100)
        self.light.setHpr(LVector3(0, -90, 0))
        render.setLight(self.light)

        plight = PointLight('plight')
        plight.setColor((0.8, 0.8, 0.8, 1))
        plnp = render.attachNewNode(plight)
        plnp.setPos(0, 0, 80)
        render.setLight(plnp)


        render.setShaderAuto()

        #self.maze2.setPos(0, 0, 10)

        #self.ballRoot.setLight(render.attachNewNode(ambientLight))
        #self.ballRoot.setLight(render.attachNewNode(directionalLight))

        # Maze 2 light
        #self.maze2.setLight(self.light)
        #self.maze2.setLight(self.ambientL)
        #self.maze2.hide()
        #self.maze.hide()

        # This section deals with adding a specular highlight to the ball to make
        # it look shiny.  Normally, this is specified in the .egg file.
        m = Material()
        m.setSpecular((1, 1, 1, 1))
        m.setShininess(96)
        self.ball.setMaterial(m, 1)

        #self.maze2.setMaterial(m,1)

        # Set maze rotation speed
        self.mazeSpeed = 10
        self.mazeVoiceSpeed = 8

        # Set maze max rotation
        self.mazeMaxRotation = 10
        self.mazeVoiceMaxRotation = 10

        # Distància minima per passar al següent punt
        self.minDist = 25
        # Pas per saltar punts del path
        self.pas = 25

        base.setBackgroundColor(0.2, 0.2, 0.2)

        self.camera2_buffer = base.win.makeTextureBuffer("c2buffer", PI_CAMERA_RES, PI_CAMERA_RES, to_ram=True)
        #mytexture.write("text1.png")
        # print(mytexture)
        # cv2.imshow("Texure1", mytexture)
        #self.camera2_buffer.setSort(-100)
        self.camera2 = base.makeCamera(self.camera2_buffer)
        self.camera2.reparentTo(render)

        self.camera2.setPosHpr(0, 0, PI_CAMERA_H, 0, -90, 0)
        self.camera2.node().getLens().setFov(PI_CAMERA_FOV)
        self.camera2.node().getLens().setFilmSize(1, 1)

        self.digitizer = Digitizer()
        self.aStar = aStar()
        self.path = None

        self.ready_to_solve = False

        # Finally, we call start for more initialization
        self.start()



    def start(self):
        # The maze model also has a locator in it for where to start the ball
        # To access it we use the find command
        #startPos = self.maze.find("**/start").getPos()
        startPos = (MAZES_START_POS[self.maze_i][0], MAZES_START_POS[self.maze_i][1], MAZE_HEIGHT + BALL_OFFSET)
        self.ballRoot.setPos(startPos)

        if not self.ready_to_solve:
            self.ballRoot.hide()

        self.pid = pid(startPos[0], startPos[1])
        # INICIALITZAR A* AMB LABERINT HARDCODEJAT, S'HA DE CANVIAR

        # ----------- self.path_matrix, self.path = self.aStar.a_star(laberint, 26, 10, 465, 448, 89, 461) -----------------

        self.indexPuntActual = 0

        self.ballV = LVector3(0, 0, 0)         # Initial velocity is 0
        self.accelV = LVector3(0, 0, 0)        # Initial acceleration is 0

        # Create the movement task, but first make sure it is not already
        # running
        taskMgr.remove("rollTask")
        self.mainLoop = taskMgr.add(self.rollTask, "rollTask")

    # This function handles the collision between the ray and the ground
    # Information about the interaction is passed in colEntry
    def groundCollideHandler(self, colEntry):
        # Set the ball to the appropriate Z value for it to be exactly on the
        # ground
        newZ = colEntry.getSurfacePoint(render).getZ()
        #self.ballRoot.setZ(newZ + .4)
        self.ballRoot.setZ(newZ + 1.4)
        # Find the acceleration direction. First the surface normal is crossed with
        # the up vector to get a vector perpendicular to the slope
        norm = colEntry.getSurfaceNormal(render)
        accelSide = norm.cross(LVector3.up())
        # Then that vector is crossed with the surface normal to get a vector that
        # points down the slope. By getting the acceleration in 3D like this rather
        # than in 2D, we reduce the amount of error per-frame, reducing jitter
        self.accelV = norm.cross(accelSide)

    # This function handles the collision between the ball and a wall
    def wallCollideHandler(self, colEntry):
        # First we calculate some numbers we need to do a reflection
        norm = colEntry.getSurfaceNormal(render) * -1  # The normal of the wall
        curSpeed = self.ballV.length()                # The current speed
        inVec = self.ballV / curSpeed                 # The direction of travel
        velAngle = norm.dot(inVec)                    # Angle of incidance
        hitDir = colEntry.getSurfacePoint(render) - self.ballRoot.getPos()
        hitDir.normalize()
        # The angle between the ball and the normal
        hitAngle = norm.dot(hitDir)

        # Ignore the collision if the ball is either moving away from the wall
        # already (so that we don't accidentally send it back into the wall)
        # and ignore it if the collision isn't dead-on (to avoid getting caught on
        # corners)
        if velAngle > 0 and hitAngle > .995:
            # Standard reflection equation
            reflectVec = (norm * norm.dot(inVec * -1) * 2) + inVec

            # This makes the velocity half of what it was if the hit was dead-on
            # and nearly exactly what it was if this is a glancing blow
            self.ballV = reflectVec * (curSpeed * (((1 - velAngle) * .5) + .5))
            # Since we have a collision, the ball is already a little bit buried in
            # the wall. This calculates a vector needed to move it so that it is
            # exactly touching the wall
            disp = (colEntry.getSurfacePoint(render) - colEntry.getInteriorPoint(render))
            newPos = self.ballRoot.getPos() + disp
            self.ballRoot.setPos(newPos)

    def rotateMaze(self, p, r):
        maxRot = self.mazeMaxRotation
        maxVel = self.mazeSpeed
        if voice_solving:
            maxRot=self.mazeVoiceMaxRotation
            maxVel=self.mazeVoiceSpeed

        if self.ready_to_solve:
            dt = globalClock.getDt()
            if r != 0 or p != 0:
                self.maze.setR(self.maze, r * maxVel * dt)

                self.maze.setP(self.maze, p * maxVel * dt)

                # Check bounds
                if self.maze.getR() > maxRot:
                    self.maze.setR(maxRot)
                elif self.maze.getR() < -maxRot:
                    self.maze.setR(-maxRot)

                if self.maze.getP() > maxRot:
                    self.maze.setP(maxRot)
                elif self.maze.getP() < -maxRot:
                    self.maze.setP(-maxRot)

                self.laso_ax.setP(self.maze.getP())


                self.maze.setH(0)
                self.laso_ax.setH(0)
                self.laso_ax.setR(0)

    def solve(self):

        # PI CAMERA PHOTO
        screenshot = self.camera2_buffer.getScreenshot()
        if screenshot:
            img = np.asarray(screenshot.getRamImage(), dtype=np.uint8).copy()
            img = img.reshape((screenshot.getYSize(), screenshot.getXSize(), 4))
            img = img[::-1]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            self.digitizer.set_src_img(img)
            self.digitizer.digitize_source()
            print("solve!!")

            trobat = False

            radi = 50
            m_seg = 30

            while trobat == False:
                trobat, self.pm, self.path = self.aStar.a_star(self.digitizer.source_mask.copy(), radi, int(m_seg), 
                                                self.digitizer.startPos[1], self.digitizer.startPos[0],
                                                self.digitizer.endPos[1], self.digitizer.endPos[0])
                radi = max(radi - 1, 0)
                m_seg = max(m_seg - 0.5, 0)
                print(radi, m_seg)

            #print(self.pm)

            self.pathFollowed = np.zeros(self.pm.shape)

            check_result = cv2.addWeighted(self.digitizer.source_img_g.astype('uint8'), 0.5,
                                           np.clip(self.pm * 255, 0, 255).astype('uint8'), 0.5, 1)
            cv2.imshow("laberint resolt sobre original", check_result)

            self.ready_to_solve = True
            self.ballRoot.show()
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

    def get_ball_position(self):
        # PI CAMERA PHOTO
        #startTime = time.time()
        screenshot = self.camera2_buffer.getScreenshot()
        if screenshot:
            #v = memoryview(screenshot.getRamImage()).tolist()
            img = np.asarray(screenshot.getRamImage(), dtype=np.uint8).copy()
            img = img.reshape((screenshot.getYSize(), screenshot.getXSize(), 4))
            img = img[::-1]
            img = img[:,:,:3]
        #self.camera2_buffer.saveScreenshot("ts.jpg")
        #img = cv2.imread("ts.jpg", 1)

        #img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        pos = self.digitizer.get_ball_pos(img)

        if pos is not False:
            pos = pos.astype(np.int32)

            self.pathFollowed[pos[0] - 1 : pos[0] + 2, pos[1] - 1 : pos[1] + 2] = 1

            img[self.pm == 1] = 0
            img[self.pm == 1, 2] = 255

            img[self.pathFollowed == 1] = 0
            img[self.pathFollowed == 1, 1] = 255

            #print(pos)

            img[pos[0] - 2 : pos[0] + 3, pos[1] - 2 : pos[1] + 3] = 0

            posActual = self.path[self.indexPuntActual]

            img[posActual[0] - 2 : posActual[0] + 3, posActual[1] - 2 : posActual[1] + 3] = 255

            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass

                #if pos is not None:
                    #print("BALL POSITION: ", pos)
                    #check_res = cv2.circle(self.digitizer.source_img, (int(pos[0]), int(pos[1])), 24, (0, 0, 255))
                    #cv2.imshow("BALL POS RES", check_res)
        return pos


    # This is the task that deals with making everything interactive
    def rollTask(self, task):
        # Standard technique for finding the amount of time since the last
        # frame
        #print("\r",self.maze.getR(), self.maze.getP(), self.ballRoot.getPos(), end="")


        dt = globalClock.getDt()
        print("\r{:.3} fps      ".format(1/dt), end="")

        # If dt is large, then there has been a # hiccup that could cause the ball
        # to leave the field if this functions runs, so ignore the frame
        if dt > .2:
            return Task.cont

        #print(action)
        if action == "start":
            a=1

        elif action == "stop":
            while action == "stop":
                a=0

       # elif action == "restart": in Line 531

        elif action == "coord":
            a=1
            #ALGUNA CRIDA A METODE DE NARCIS/MARC

        key_down = base.mouseWatcherNode.is_button_down

        if self.ready_to_solve:

            if key_down(KeyboardButton.ascii_key('d')):
                screenshot = self.camera2_buffer.getScreenshot()
                if screenshot:
                    v = memoryview(screenshot.getRamImage()).tolist()
                    img = np.array(v, dtype=np.uint8)
                    img = img.reshape((screenshot.getYSize(), screenshot.getXSize(), 4))
                    img = img[::-1]
                    #self.digitizer.set_src_img(img)
                    #self.digitizer.digitalize_source()
                    cv2.imshow('img', img)
                    #cv2.waitKey(0)

            if key_down(KeyboardButton.ascii_key('s')):
                print("Screenshot!")
                self.camera2_buffer.saveScreenshot("screenshot.jpg")


            # The collision handler collects the collisions. We dispatch which function
            # to handle the collision based on the name of what was collided into
            for i in range(self.cHandler.getNumEntries()):
                entry = self.cHandler.getEntry(i)
                name = entry.getIntoNode().getName()
                if action == "restart":
                    self.loseGame(entry)
                if name == "wall_col":
                    self.wallCollideHandler(entry)
                elif name == "ground_col":
                    self.groundCollideHandler(entry)
                elif name == "loseTrigger":
                    vr.restart=1
                    global th
                    th = threading.Thread(target=listenVoice)
                    th.start()
                    self.loseGame(entry)

            # Read the mouse position and tilt the maze accordingly
            # Rotation axes use (roll, pitch, heave)
            """
            if base.mouseWatcherNode.hasMouse():
                mpos = base.mouseWatcherNode.getMouse()  # get the mouse position
                self.maze.setP(mpos.getY() * -10)
                self.maze.setR(mpos.getX() * 10)
            """


            ballPos = self.get_ball_position()
            #print("BALL POS: ", ballPos)

            posFPixel = self.path[self.indexPuntActual]

            xFinal = posFPixel[1] #posFPixel[1]/np.shape(laberint)[0]*13 - 6.5
            yFinal = posFPixel[0] #-(posFPixel[0]/np.shape(laberint)[1]*13.5 - 6.8)

            dist = math.sqrt((xFinal - ballPos[1])**2 + (yFinal - ballPos[0])**2)

            if(dist < self.minDist):
                if(self.indexPuntActual == len(self.path) - 1):
                    print("SOLVED!!", end = "")

                while(self.aStar.distance((ballPos[0], ballPos[1]), self.path[self.indexPuntActual]) < self.pas):
                    if(self.indexPuntActual < len(self.path) - 1):
                        self.indexPuntActual += 1
                    else:
                        break

            # ball pos (y,x)
            
            #print("END POS: ", self.digitizer.endPos)

            if voice_solving:
                p_rotation = dir_veu[0]
                r_rotation = dir_veu[1]
                if p_rotation == 0 and r_rotation == 0 and ballPos is not None:
                     p_rotation, r_rotation = self.pid.getPR(ballPos[1], ballPos[0], ballPos[1], ballPos[0], self.maze.getP(), self.maze.getR(), dt)

            else:
                p_rotation = 0
                r_rotation = 0
                #print(ballPos, dist)


                #print(ballPos)

                if ballPos is not None:

                    p_rotation, r_rotation = self.pid.getPR(ballPos[1], ballPos[0],
                        xFinal, yFinal,
                        self.maze.getP(), self.maze.getR(), dt)
                    #print(p_rotation, r_rotation)

            if key_down(KeyboardButton.up()):
                p_rotation = -1
            elif key_down(KeyboardButton.down()):
                p_rotation = 1

            if key_down(KeyboardButton.left()):
                r_rotation = -1
            elif key_down(KeyboardButton.right()):
                r_rotation = 1


            self.rotateMaze(p_rotation, r_rotation)



            # Finally, we move the ball
            # Update the velocity based on acceleration
            self.ballV += self.accelV * dt * ACCEL
            # Clamp the velocity to the maximum speed
            if self.ballV.lengthSquared() > MAX_SPEED_SQ:
                self.ballV.normalize()
                self.ballV *= MAX_SPEED
            # Update the position based on the velocity
            self.ballRoot.setPos(self.ballRoot.getPos() + (self.ballV * dt))
            #print(self.ballRoot.getPos())

            # This block of code rotates the ball. It uses something called a quaternion
            # to rotate the ball around an arbitrary axis. That axis perpendicular to
            # the balls rotation, and the amount has to do with the size of the ball
            # This is multiplied on the previous rotation to incrimentally turn it.
            prevRot = LRotationf(self.ball.getQuat())
            axis = LVector3.up().cross(self.ballV)
            newRot = LRotationf(axis, 45.5 * dt * self.ballV.length())
            self.ball.setQuat(prevRot * newRot)

        elif key_down(KeyboardButton.ascii_key('1')):
            self.solve()

        if key_down(KeyboardButton.ascii_key('i')):
            self.light.setY(self.light.getY() + 10 * dt)
        elif key_down(KeyboardButton.ascii_key('k')):
            self.light.setY(self.light.getY() - 10 * dt)

        if key_down(KeyboardButton.ascii_key('j')):
            self.light.setX(self.light.getX() - 10 * dt)
        elif key_down(KeyboardButton.ascii_key('l')):
            self.light.setX(self.light.getX() + 10 * dt)

        if key_down(KeyboardButton.ascii_key('u')):
            self.lightColor += 10000 * dt
            self.light.node().setColor((self.lightColor, self.lightColor, self.lightColor, 1))
        elif key_down(KeyboardButton.ascii_key('o')):
            self.lightColor -= 10000 * dt
            self.light.node().setColor((self.lightColor, self.lightColor, self.lightColor, 1))

        if key_down(KeyboardButton.ascii_key('8')):
            self.alightColor += 1 * dt
            self.ambientL.node().setColor((self.alightColor, self.alightColor, self.alightColor, 1))
        elif key_down(KeyboardButton.ascii_key('9')):
            self.alightColor -= 1 * dt
            self.ambientL.node().setColor((self.alightColor, self.alightColor, self.alightColor, 1))

        if key_down(KeyboardButton.ascii_key('r')):
            base.trackball.node().set_pos(0, 200, 0)
            base.trackball.node().set_hpr(0, 60, 0)

        return Task.cont       # Continue the task indefinitely

    # If the ball hits a hole trigger, then it should fall in the hole.
    # This is faked rather than dealing with the actual physics of it.
    def loseGame(self, entry):
        # The triggers are set up so that the center of the ball should move to the
        # collision point to be in the hole
        global action
        action = "start"
        toPos = entry.getInteriorPoint(render)
        taskMgr.remove('rollTask')  # Stop the maze task
        self.pathFollowed = np.zeros(self.pm.shape)
        self.maze.setP(0)
        self.maze.setR(0)

        # Move the ball into the hole over a short sequence of time. Then wait a
        # second and call start to reset the game
        Sequence(
            Parallel(
                LerpFunc(self.ballRoot.setX, fromData=self.ballRoot.getX(),
                         toData=toPos.getX(), duration=.1),
                LerpFunc(self.ballRoot.setY, fromData=self.ballRoot.getY(),
                         toData=toPos.getY(), duration=.1),
                LerpFunc(self.ballRoot.setZ, fromData=self.ballRoot.getZ(),
                         toData=self.ballRoot.getZ() - .9, duration=.2)),
            Wait(1),
            Func(self.start)).start()

    def toggleInterval(self, ival):
        if ival.isPlaying():
            ival.pause()
        else:
            ival.resume()

    def shaderSupported(self):
        return base.win.getGsg().getSupportsBasicShaders() and \
               base.win.getGsg().getSupportsDepthTexture() and \
               base.win.getGsg().getSupportsShadowFilter()

# Finally, create an instance of our class and start 3d rendering
demo = BallInMazeDemo()

try:

    th = threading.Thread(target=listenVoice, daemon = True)
    th.start()

except:
    print("Error: unable to start thread")


demo.run()