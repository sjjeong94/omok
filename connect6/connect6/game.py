import os
import time
import json
import numpy as np
import pygame
from urllib import request
from connect6 import Connect6
from connect6.version import VERSION

url = "https://raw.githubusercontent.com/sjjeong94/connect6/main/connect6_assets/"
download_links = {
    "board": url + "board.png",
    "black": url + "stone_black.png",
    "white": url + "stone_white.png",
}

images_path = "./connect6_assets"
images = {
    "board": os.path.join(images_path, "board.png"),
    "black": os.path.join(images_path, "stone_black.png"),
    "white": os.path.join(images_path, "stone_white.png"),
}


def check_images():
    if not os.path.exists(images_path):
        os.makedirs(images_path, exist_ok=True)
    if not os.path.exists(images['board']):
        request.urlretrieve(download_links['board'], images['board'])
    if not os.path.exists(images['black']):
        request.urlretrieve(download_links['black'], images['black'])
    if not os.path.exists(images['white']):
        request.urlretrieve(download_links['white'], images['white'])


SIZE = 19
BOARD_SIZE = 800
STONE_SIZE = 40
B0 = STONE_SIZE // 2
B1 = BOARD_SIZE - B0


class Connect6Game:
    def __init__(self):
        check_images()
        pygame.init()
        pygame.display.set_caption("Connect6")
        self.game_pad = pygame.display.set_mode((1100, 800))
        self.board = pygame.image.load(images['board'])
        self.stone_black = pygame.image.load(images['black'])
        self.stone_white = pygame.image.load(images['white'])
        self.clock = pygame.time.Clock()
        self.fontObj = pygame.font.Font(None, 28)

        self.env = Connect6()

    def display(self):
        self.display_board()
        self.display_point()
        self.display_moves()
        self.display_prev()
        self.display_text()
        pygame.display.flip()
        self.clock.tick(60)

    def display_board(self):
        self.game_pad.fill((0, 0, 0))
        self.game_pad.blit(self.board, (0, 0))

    def display_moves(self):
        state = self.env.get_state()
        move_history = self.env.get_move_history()
        for i in range(len(move_history)):
            move = move_history[i]
            y, x = divmod(move, SIZE)
            stone = state[y, x]
            if stone != 0:
                if stone == 1:
                    src = self.stone_black
                else:
                    src = self.stone_white
                ax = B0 + x*STONE_SIZE
                ay = B0 + y*STONE_SIZE
                self.game_pad.blit(src, (ax, ay))
                t = '%d' % i
                r = self.fontObj.render(t, True, (192, 0, 96))
                self.game_pad.blit(r, (ax+3, ay+10))

    def display_point(self):
        mx, my = pygame.mouse.get_pos()
        if mx >= B0 and my >= B0 and mx <= B1 and my <= B1:
            x = np.clip((mx - B0)//STONE_SIZE, 0, SIZE-1)
            y = np.clip((my - B0)//STONE_SIZE, 0, SIZE-1)
            if self.env.get_player() == 1:
                color = (0, 0, 0)
            else:
                color = (255, 255, 255)
            pygame.draw.rect(self.game_pad, color,
                             (STONE_SIZE*(x+1)-10, STONE_SIZE*(y+1)-10, 20, 20))

    def display_prev(self):
        move_history = self.env.get_move_history()
        if len(move_history) > 0:
            move = move_history[-1]
            y, x = divmod(move, SIZE)
            color = (0, 255, 128)
            pygame.draw.circle(
                self.game_pad, color, (STONE_SIZE*(x+1), STONE_SIZE*(y+1)), STONE_SIZE//2+2, 2)

    def display_text(self):
        text = [
            'Player %d' % self.env.get_player(),
            'Move %d' % len(self.env.get_move_history()),
            'Winner %d' % self.env.get_winner(),
            '',
            '< Key Control >',
            '[Space] reset',
            '[B] move back',
            '',
        ]
        p = [810, 10]
        for t in text:
            r = self.fontObj.render(t, True, (0, 255, 128))
            self.game_pad.blit(r, p)
            p[1] += 40

        t = 'version %s' % VERSION
        r = self.fontObj.render(t, True, (0, 255, 128))
        self.game_pad.blit(r, [810, 770])

    def get_move(self):
        mx, my = pygame.mouse.get_pos()
        if mx >= B0 and my >= B0 and mx <= B1 and my <= B1:
            x = np.clip((mx - B0)//STONE_SIZE, 0, STONE_SIZE-1)
            y = np.clip((my - B0)//STONE_SIZE, 0, STONE_SIZE-1)
            action = y*SIZE + x
            self.env.move(action)

    def process_event(self):
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_button = pygame.mouse.get_pressed()
                if mouse_button[0]:
                    self.get_move()
            if event.type == pygame.KEYDOWN:
                key_button = pygame.key.get_pressed()
                if key_button[32]:
                    self.env.reset()
                if key_button[ord('l')] or key_button[ord('L')]:
                    print(self.env.get_log())
                if key_button[ord('s')] or key_button[ord('S')]:
                    os.makedirs('logs', exist_ok=True)
                    file_name = 'logs/%d.json' % int(time.time())
                    with open(file_name, 'w') as f:
                        json.dump(self.env.get_log(), f, separators=(',', ':'))
                if key_button[ord('b')] or key_button[ord('B')]:
                    self.env.move_back()
                if key_button[ord('o')] or key_button[ord('O')]:
                    print(self.env)

        return running

    def __call__(self):
        return self.call()

    def call(self):
        self.display()
        return self.process_event()

    def __del__(self):
        pygame.quit()
