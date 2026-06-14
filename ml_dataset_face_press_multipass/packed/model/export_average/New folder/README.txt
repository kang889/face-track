RIGHT CHEEK HEIGHT MAP README

HOW I GOT THIS DATA
I recorded repeated finger presses on the same right-cheek region.
The cheek motion was tracked using 25 selected cheek points.
Clean samples were filtered from the recording.
A model was trained to predict how those 25 cheek points move during the press.

HOW I TURNED IT INTO A HEIGHT MAP
For the clean samples, I took the predicted depth-like movement of the 25 cheek points.
I averaged those point values.
Then I filled the space between the 25 points to create one smooth regional cheek map.
So this height map is an approximate cheek patch, not a true dense depth scan.

WHAT THE PNG IS
regional_heightmap_pred.png
This is the main output.
It is the approximate right-cheek height map image for testing in the renderer.

WHAT THE CSV IS
regional_heightmap_pred_float.csv
This is the numeric version of the same height map.
It contains the height value for each pixel in the cheek patch.

WHAT THE MP4 IS
recording_example.mp4
This is an example of how the data was recorded.
It shows the real right-cheek region used during capture.
