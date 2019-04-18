import unittest
import math
import vs


class TestIMUStabilization(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def checkAlmostEqualQuaternions(self, q1, q2):
        self.assertAlmostEqual(q1.getQ0(), q2.getQ0(), 5)
        self.assertAlmostEqual(q1.getQ1(), q2.getQ1(), 5)
        self.assertAlmostEqual(q1.getQ2(), q2.getQ2(), 5)
        self.assertAlmostEqual(q1.getQ3(), q2.getQ3(), 5)

    def testFusionIMU(self):
        acc = vs.Vector(0, 0, 1)
        gyr = vs.Vector(0, 0, 0)
        fusionIMU = vs.FusionIMU()
        q = fusionIMU.init(acc, 0)
        qTarget = vs.Quat(1., 0, 0, 0)
        self.checkAlmostEqualQuaternions(q, qTarget)

        angle = math.pi / 3

        acc = vs.Vector(0, math.sin(angle), math.cos(angle))
        q = fusionIMU.init(acc, 0)
        qTarget = vs.Quat(math.cos(angle/2.), math.sin(angle/2.), 0, 0)
        self.checkAlmostEqualQuaternions(q, qTarget)

        acc = vs.Vector(0, -math.sin(angle), math.cos(angle))
        q = fusionIMU.init(acc, 0)
        qTarget = vs.Quat(math.cos(angle/2.), -math.sin(angle/2.), 0, 0)
        self.checkAlmostEqualQuaternions(q, qTarget)

        acc = vs.Vector(math.sin(angle), 0, math.cos(angle))
        q = fusionIMU.init(acc, 0)
        qTarget = vs.Quat(math.cos(angle/2.), 0, -math.sin(angle/2.), 0)
        self.checkAlmostEqualQuaternions(q, qTarget)

        acc = vs.Vector(-math.sin(angle), 0, math.cos(angle))
        q = fusionIMU.init(acc, 0)
        qTarget = vs.Quat(math.cos(angle/2.), 0, math.sin(angle/2.), 0)
        self.checkAlmostEqualQuaternions(q, qTarget)


        # use accelerometer only
        fusionIMU.setFusionFactor(1)  # discard gyroscope during fusion
        acc = vs.Vector(0, 0, 1)
        q = fusionIMU.init(acc, 1000000)
        qTarget = vs.Quat(1, 0, 0, 0)
        self.checkAlmostEqualQuaternions(q, qTarget)

        acc = vs.Vector(0, math.sin(angle), math.cos(angle))
        gyr = vs.Vector(1, 1, 1)
        q = fusionIMU.update(gyr, acc, 2000000)
        qTarget = vs.Quat(math.cos(angle/2.), math.sin(angle/2.), 0, 0)
        self.checkAlmostEqualQuaternions(q, qTarget)


        # use gyroscope only
        fusionIMU.setFusionFactor(0)  # discard accelerometer during fusion
        acc = vs.Vector(0, 0, 1)
        q = fusionIMU.init(acc, 1000000)
        qTarget = vs.Quat(1, 0, 0, 0)
        self.checkAlmostEqualQuaternions(q, qTarget)

        acc = vs.Vector(0, 0, 1)
        gyr = vs.Vector(angle, 0, 0)
        q = fusionIMU.update(gyr, acc, 1033333)  # integrate only during 1/30 of a second
        qTarget = vs.Quat(math.cos(angle/(2.*30.)), math.sin(angle/(2.*30.)), 0, 0)
        self.checkAlmostEqualQuaternions(q, qTarget)

