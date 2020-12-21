# Python Server

Analyzes signals from Myo and controls Mandro.

## Mandro control API Spec
  * Commuicate with SocketServer class.
  * Commands are describe in 1 byte integer.
  * 6th bit (from lower bit) means grab or release.
    * Set 1 to grab finger.
    * Set 0 to release finger.
  * 1~5th bit (from lower bit) means which finger to command.
    * Set 1 to target i-th finger.
### Example:
*  Grap thumb(1st finger) and index finger(2nd finger).
   *  Send 35. (Which means `100011` in binary.)
* Release middle finger(3th finger) and pinky(5th finger).
  * Send 20. (Which means `010100` in binary.)