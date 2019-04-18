Feature: RTMP Client

  Trying to open two RTMP connections to an RTMP server which does not exist.
  After being unable to create a connection, videostitch-cmd should exit with
  an error code.

  If there is a crash, the error code may be different.

    Scenario: RTMP Client with invalid inputs
        When  I launch videostitch-cmd with rtmp/rtmp_client_invalid.ptv and "-d 0 -f 0 -l 100"
        Then  I expect the command to fail with code 1

    Scenario: Test the client (we will get usefull information from ASAN/TSAN)
        Given There is an RTMP flow
        And   I generated rtmp_client.ptv with rtmp/rtmp_client.tpl
        When  I launch videostitch-cmd with ../data/rtmp_client.ptv from repo and "-d 0 -f 0 -l 100"
        Then  I expect the command to succeed

    @slow
    Scenario: Input stream stops and re-starts
        Given There is an RTMP flow
        And   I generated rtmp_client.ptv with rtmp/rtmp_client.tpl
        When  I launch videostitch-cmd with ../data/rtmp_client.ptv and "-d 0 -f 0 -l 9" in the background
        And   I wait the RTMP flow to stop
        And   I start the RTMP flow
        Then  The background process was successful

