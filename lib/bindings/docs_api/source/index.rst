StitchingBox's API documentation!
============================================

API documentation for the StitchingBox server.

Contents:

.. toctree::
   :titlesonly:

   errors
   enumerators
   boxapi
   stitcherapi
   osapi
   cameraapi
   releasenotes

Discovery and Connection
=========================
The discovery and connection to the Access Point (AP) process must be performed as a first step.

1. Connect to the AP of the Box named: **vsstitchingboxXYZ**
2. Use the standard password: **5t1tch1ngb0x**
3. Once connected to the AP, the calls to the API must be directed as a POST messages.
4. The API endpoint **http://127.0.0.10:8877/commands/execute**

POST Message formats
=====================
For calling the API, the client must send a message to the server in the format:

.. code-block:: json

	"name": module.funtion,
	"parameters": {
		"param1": value1,
		"param2": value2,
		...
	}


Successfull response of the request will have the form:

.. code-block:: json	

    "result": some human readable message

Error message will be:

.. code-block:: json

	"error": {
		"message": some human readable message,
		"code": ERROR_CODE
	}


First Call
==========

Once the client is connected to the access point the first logic step is to get the information:

.. code-block:: json

	"name": 'stitcher.get_config'

Status and Keep alive
=====================

To ensure that the server and the client are on sync, the client should use the 'stitcher.status' API call in order to show the current server state.
Poll the same status request to make sure the server is still responding. After X tries without response, the client should show the error.