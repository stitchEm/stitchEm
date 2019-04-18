import os
import tempfile
import unittest
import errors

from system.wifi_manager import wifi_manager


SHORT_PASSWD = "123"
LONG_PASSWD = "0123456789012345678901234567890123456789012345678901234567890123456789"
NEW_PASSWD = "1234567890"
NEW_PASSWD_2 = "123456789022"
WRONG_SSID = "0123456789012345678901234567890123456789"



class TestWifiManager(unittest.TestCase):
    
    def setUp(self):
        
        found_ssid = False
        found_chan = False
        found_passwd = False
        
        with open(wifi_manager.HOSTAPD_FILE) as file:
            for line in file:
                if "channel=" in line:
                    self.orig_chan = line.replace("channel=", "") 
                    self.orig_chan = self.orig_chan.rstrip("\n")
                    found_chan = True
                if "ssid=" in line:
                    self.orig_ssid = line.replace("ssid=", "")
                    self.orig_ssid = self.orig_ssid.rstrip("\n")
                    found_ssid = True
                if "wpa_passphrase=" in line:
                    self.orig_passwd = line.replace("wpa_passphrase=", "")
                    self.orig_passwd = self.orig_passwd.rstrip("\n")
                    found_passwd = True
                     
                    
        self.assertTrue(found_chan ,"Wifi conf test setup : can't find channel") 
        self.assertTrue(found_ssid ,"Wifi conf test setup : can't find ssid")
        self.assertTrue(found_passwd ,"Wifi conf test setup : can't find passwd")

	print "Wifi conf test : original SSID : " + self.orig_ssid
	print "Wifi conf test : original channel : " + self.orig_chan
	print "Wifi conf test : original password : "+ self.orig_passwd
    
    def testBadParameters(self):
        with self.assertRaises(errors.WrongWifiPassword):
            wifi_manager.set_wifi_conf(SHORT_PASSWD, "kingklem")
            
        with self.assertRaises(errors.WrongWifiPassword):
            wifi_manager.set_wifi_conf(LONG_PASSWD, "kingklem")
            
        with self.assertRaises(errors.WifiError):    
            wifi_manager.set_wifi_conf(self.orig_passwd, WRONG_SSID)
            


        with self.assertRaises(errors.WrongWifiPassword):       
            wifi_manager.set_wifi_conf(SHORT_PASSWD, new_channel="4")

        with self.assertRaises(errors.WrongWifiPassword):
            wifi_manager.set_wifi_conf(LONG_PASSWD, new_channel="4")

        with self.assertRaises(errors.WifiError):   
            wifi_manager.set_wifi_conf(self.orig_passwd, new_channel="19")

        with self.assertRaises(errors.WifiError): 
            wifi_manager.set_wifi_conf(self.orig_passwd, new_channel="123")


        with self.assertRaises(errors.WrongWifiPassword): 
            wifi_manager.set_wifi_conf(SHORT_PASSWD, new_passwd=NEW_PASSWD)
            
        with self.assertRaises(errors.WrongWifiPassword):
            wifi_manager.set_wifi_conf(LONG_PASSWD,  new_passwd=NEW_PASSWD)
            
        with self.assertRaises(errors.WifiError): 
            wifi_manager.set_wifi_conf(self.orig_passwd,  new_passwd=SHORT_PASSWD)
            
        with self.assertRaises(errors.WifiError): 
            wifi_manager.set_wifi_conf(self.orig_passwd,  new_passwd=LONG_PASSWD) 

            
        with self.assertRaises(errors.WrongWifiPassword): 
            wifi_manager.set_wifi_conf(SHORT_PASSWD, "king klem", "4", NEW_PASSWD)
            
        with self.assertRaises(errors.WrongWifiPassword): 
            wifi_manager.set_wifi_conf(LONG_PASSWD, "king klem", "4", NEW_PASSWD)
            
        with self.assertRaises(errors.WifiError): 
            wifi_manager.set_wifi_conf(self.orig_passwd, WRONG_SSID, "4", NEW_PASSWD)
            
        with self.assertRaises(errors.WifiError): 
            wifi_manager.set_wifi_conf(self.orig_passwd, "king klem", "19", NEW_PASSWD)
  
        with self.assertRaises(errors.WifiError): 
            wifi_manager.set_wifi_conf(self.orig_passwd, "king klem", "123", NEW_PASSWD)
            
        with self.assertRaises(errors.WifiError): 
            wifi_manager.set_wifi_conf(self.orig_passwd, "king klem", "4", SHORT_PASSWD)
            
        with self.assertRaises(errors.WifiError): 
            wifi_manager.set_wifi_conf(self.orig_passwd, "king klem", "4", LONG_PASSWD)
 

    
    def testGetVal(self):
        
        found = False
        ssid = wifi_manager.get_ssid()
        with open(wifi_manager.HOSTAPD_FILE) as file:
            for line in file:
                if "ssid=" in line:
                    orig_val_ssid = line.replace("ssid=", "")
                    orig_val_ssid = orig_val_ssid.rstrip("\n")
                    self.assertEqual(ssid, orig_val_ssid, "get wrong ssid")
                    found = True 
            
                
        self.assertTrue(found,"Wifi conf test : can't find ssid")                       
        
        found = False
        chan = wifi_manager.get_channel()
        with open(wifi_manager.HOSTAPD_FILE) as file:
            for line in file:
                if "channel" in line :
                    orig_val_chan = line.replace("channel=", "")
                    orig_val_chan = orig_val_chan.rstrip("\n")
                    self.assertEqual(chan, orig_val_chan, "get wrong channel")
                    found = True

        self.assertTrue(found,"Wifi conf test : can't find channel")    

        
    def testSetConf(self):
        wifi_manager.set_wifi_conf(self.orig_passwd, "test_test", "4", NEW_PASSWD)
        
        found_ssid = False
        found_chan = False
        found_passwd = False
        
        with open(wifi_manager.HOSTAPD_FILE) as file:
            for line in file:
                if "channel=" in line:
                    orig_val_chan = line.replace("channel=", "")
                    orig_val_chan = orig_val_chan.rstrip("\n")
                    self.assertEqual(orig_val_chan, "4", "Wifi conf test : get wrong channel")
                    found_chan = True
                if "ssid=" in line:
                    orig_val_ssid = line.replace("ssid=", "")
                    orig_val_ssid = orig_val_ssid.rstrip("\n")
                    self.assertEqual(orig_val_ssid, "ORAH4i_test_test", "Wifi conf test : get wrong ssid")
                    found_ssid = True
                if "wpa_passphrase=" in line:
                    orig_val_passwd = line.replace("wpa_passphrase=", "")
                    orig_val_passwd = orig_val_passwd.rstrip("\n")
                    self.assertEqual(orig_val_passwd, NEW_PASSWD, "Wifi conf test : get wrong password")
                    found_passwd = True
                    
                    
        self.assertTrue(found_chan ,"Wifi conf test : can't find channel") 
        self.assertTrue(found_ssid ,"Wifi conf test : can't find ssid")
        self.assertTrue(found_passwd ,"Wifi conf test : can't find passwd")
        
        #check each set function
        wifi_manager.set_wifi_conf(NEW_PASSWD, "kingklem")
        wifi_manager.set_wifi_conf(NEW_PASSWD, new_channel="3")
        wifi_manager.set_wifi_conf(NEW_PASSWD, new_passwd=NEW_PASSWD_2)
        
        
        with open(wifi_manager.HOSTAPD_FILE) as file:
            for line in file:
                if "channel=" in line:
                    orig_val_chan = line.replace("channel=", "")
                    orig_val_chan = orig_val_chan.rstrip("\n")
                    self.assertEqual(orig_val_chan, "3", "Wifi conf test : get wrong channel")
                    found_chan = True
                if "ssid=" in line:
                    orig_val_ssid = line.replace("ssid=", "")
                    orig_val_ssid = orig_val_ssid.rstrip("\n")
                    self.assertEqual(orig_val_ssid, "ORAH4i_kingklem", "Wifi conf test : get wrong ssid")
                    found_ssid = True
                if "wpa_passphrase=" in line:
                    orig_val_passwd = line.replace("wpa_passphrase=", "")
                    orig_val_passwd = orig_val_passwd.rstrip("\n")
                    self.assertEqual(orig_val_passwd, NEW_PASSWD_2, "Wifi conf test : get wrong password")
                    found_passwd = True
                    
            
        self.assertTrue(found_chan ,"Wifi conf test : can't find channel") 
        self.assertTrue(found_ssid ,"Wifi conf test : can't find ssid")
        self.assertTrue(found_passwd ,"Wifi conf test : can't find passwd")
        
        #put back original value
        wifi_manager.set_wifi_conf(NEW_PASSWD_2, self.orig_ssid, self.orig_chan, self.orig_passwd)
                    
        
        with open(wifi_manager.HOSTAPD_FILE) as file:
            for line in file:
                if "channel=" in line:
                    orig_val_chan = line.replace("channel=", "")
                    orig_val_chan = orig_val_chan.rstrip("\n")
                    self.assertEqual(orig_val_chan, self.orig_chan, "Wifi conf test : get wrong channel")
                    found_chan = True
                if "ssid=" in line:
                    orig_val_ssid = line.replace("ssid=", "")
                    orig_val_ssid = orig_val_ssid.rstrip("\n")
                    self.assertEqual(orig_val_ssid, self.orig_ssid, "Wifi conf test : get wrong ssid")
                    found_ssid = True
                if "wpa_passphrase=" in line:
                    orig_val_passwd = line.replace("wpa_passphrase=", "")
                    orig_val_passwd = orig_val_passwd.rstrip("\n")
                    self.assertEqual(orig_val_passwd, self.orig_passwd, "Wifi conf test : get wrong password")
                    found_passwd = True


        self.assertTrue(found_chan ,"Wifi conf test : can't find channel")
        self.assertTrue(found_ssid ,"Wifi conf test : can't find ssid")
        self.assertTrue(found_passwd ,"Wifi conf test : can't find passwd")

        
