from matrix_client.client import MatrixClient
from matrix_client.api import MatrixHttpApi
import json

# Replace these values with your own
MATRIX_SERVER = "https://matrix.org"
BOT_USERNAME = "dream_bot_1"
# Open the file and read its contents
file_path = '/data/common/configs/bot/dream_bot_psw'
with open(file_path, 'r') as file:
    file_contents = file.read()
# BOT_PASSWORD = "dream_bot_1password"
BOT_PASSWORD = file_contents
TARGET_USER_ID = "@ranson.ad:matrix.org"
MESSAGE = "Hello from the bot!"

class CustomMatrixHttpApi(MatrixHttpApi):
    def get_room_members(self, room_id):
        """Get the list of members in the given room."""
        content = self._send("GET", f"/rooms/{room_id}/members")
        return content
    
def main(target_user,msg,group=''):
    try:
        # convert ubuntu username to element username
        target_user = lookup_user(target_user)

        # Create a client instance
        client = MatrixClient(MATRIX_SERVER)

        # Log in with the bot account
        client.login_with_password(username=BOT_USERNAME, password=BOT_PASSWORD)

        # Get the access token and create an API instance
        access_token = client.api.token
        api = CustomMatrixHttpApi(MATRIX_SERVER, token=access_token)

        # Get a list of rooms the bot is in
        rooms = client.get_rooms()
        msg_sent = False
        if group == '':
            # Check if there's already a private room with the target user
            target_room = None
            for room_id, room in rooms.items():
                members = room.get_joined_members()
                # make dict of all members
                all_members = []
                for iMem in range(len(members)):
                    all_members.append(members[iMem].user_id)
                if target_user in all_members and len(members) == 2:
                    target_room = room    
                    # Send the message
                    target_room.send_text(msg)
                    msg_sent = True
                    break
        else:
            # try to find the group requested !byLPEQmNPNnuSYFYuP:matrix.org
            for room_id, room in rooms.items():
                display_name = room.display_name
                if display_name == group:
                    room.send_text(msg)
                    msg_sent = True
        
        if msg_sent == False
            print("WARNING: YOU DO NOT HAVE AN ELEMENT USERNAME PAIRED TO YOUR UBUNTU USERNAME - PLEASE REQUEST THIS FOR ELEMENT NOTIFICATIONS")
            
        # If no private room exists, create one and invite the target user
        # if not target_room:
        #     target_room = client.create_room(is_public=False, invitees=[target_user])
        #     target_room.send_text('Hi, I''m the friendly server bot who will tell you when your dream server jobs are complete!')

        # Log out and close the connection
        client.logout()
    except:
        print('Error sending matrix msg')

def lookup_user(username):
    if username == 'adamranson':
        return '@ranson.ad:matrix.org'
    elif username == 'melinatimplalexi':
        return '@melina_timplalexi:matrix.org'
    elif username == 'pmateosaparicio':
        return '@pmateosaparicio:matrix.org'
    elif username == 'antoniofernandez':
        return '@boxerito:matrix.org'
    elif username == 'rubencorreia':
        return '@rubencorreia:matrix.org'    
    elif username == 'sebastianrodriguez':
        return '@sebastian.rdz:matrix.org'
    else:
        return ''
    

if __name__ == "__main__":
    main()
