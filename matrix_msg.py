from matrix_client.client import MatrixClient
from matrix_client.api import MatrixHttpApi
import json
import time
from datetime import datetime, timedelta


# Replace these values with your own
MATRIX_SERVER = "https://matrix.org"
BOT_USERNAME = "dream_bot_1"
password_file_path = '/data/common/configs/bot/dream_bot_psw'
token_file_path = '/data/common/configs/bot/token.txt'

class CustomMatrixHttpApi(MatrixHttpApi):
    def get_room_members(self, room_id):
        """Get the list of members in the given room."""
        content = self._send("GET", f"/rooms/{room_id}/members")
        return content

def save_token(token):
    with open(token_file_path, 'w') as file:
        file.write(token)

def load_token():
    try:
        with open(token_file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

def login_and_save_token():
    print("Making token")
    with open(password_file_path, 'r') as file:
        bot_password = file.read()
    
    client = MatrixClient(MATRIX_SERVER)
    client.login_with_password(username=BOT_USERNAME, password=bot_password)
    save_token(client.api.token)
    return client.api.token

def main(target_user, msg, group=''):
    try:
        token = load_token()
        if not token:
            token = login_and_save_token()
            
        try:
            client = MatrixClient(MATRIX_SERVER, token=token)
            api = CustomMatrixHttpApi(MATRIX_SERVER, token=token)
        except:
            token = login_and_save_token()
            try:
                client = MatrixClient(MATRIX_SERVER, token=token)
                api = CustomMatrixHttpApi(MATRIX_SERVER, token=token)
            except:
                print('Error: could not login to matrix')          
            

        target_user = lookup_user(target_user)
        rooms = client.get_rooms()
        msg_sent = False

        if group == '':
            # default to server queue notifications channel
            group = 'Server queue notifications'

        if group == '':
            target_room = None
            for room_id, room in rooms.items():
                members = room.get_joined_members()
                all_members = [member.user_id for member in members]
                if target_user in all_members and len(members) == 2:
                    target_room = room
                    target_room.send_text(msg)
                    msg_sent = True
                    break
        else:
            for room_id, room in rooms.items():
                if room.display_name == group:
                    room.send_text(msg)
                    msg_sent = True

        if not msg_sent:
            print("WARNING: YOU DO NOT HAVE AN ELEMENT USERNAME PAIRED TO YOUR UBUNTU USERNAME - PLEASE REQUEST THIS FOR ELEMENT NOTIFICATIONS")
    except:
        print('####### UNHANDLED ELEMENT ERROR ##############')

   

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
    main("some_user", "Hello from the bot!")
