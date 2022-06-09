import os
import time
import json
import numpy as np
import pygame
from urllib import request
from omok import Omok

download_links = {
    "board": "https://raw.githubusercontent.com/sjjeong94/omok/main/images/board.png",
    "black": "https://raw.githubusercontent.com/sjjeong94/omok/main/images/stone_black.png",
    "white": "https://raw.githubusercontent.com/sjjeong94/omok/main/images/stone_white.png",
}

images = {
    "board": "./omok_images/board.png",
    "black": "./omok_images/stone_black.png",
    "white": "./omok_images/stone_white.png",
}


def check_images():
    if not os.path.exists('./omok_images'):
        os.makedirs('./omok_images', exist_ok=True)
    if not os.path.exists(images['board']):
        request.urlretrieve(download_links['board'], images['board'])
    if not os.path.exists(images['black']):
        request.urlretrieve(download_links['black'], images['black'])
    if not os.path.exists(images['white']):
        request.urlretrieve(download_links['white'], images['white'])


class OmokGame:
    def __init__(self):
        check_images()
        pygame.init()
        pygame.display.set_caption("OMOK")
        self.game_pad = pygame.display.set_mode((1100, 800))
        self.board = pygame.image.load(images['board'])
        self.stone_black = pygame.image.load(images['black'])
        self.stone_white = pygame.image.load(images['white'])
        self.clock = pygame.time.Clock()
        self.fontObj = pygame.font.Font(None, 32)

        self.env = Omok()

    def display(self):
        self.display_board()
        self.display_point()
        self.display_move()
        self.display_text()
        pygame.display.flip()
        self.clock.tick(60)

    def display_board(self):
        self.game_pad.fill((0, 0, 0))
        self.game_pad.blit(self.board, (0, 0))
        state = self.env.get_state()
        for i in range(len(self.env.get_move_history())):
            move = self.env.get_move_history()[i]
            y = move // 15
            x = move % 15
            stone = state[y, x]
            if stone != 0:
                if stone == 1:
                    src = self.stone_black
                else:
                    src = self.stone_white
                ax = 25 + x*50
                ay = 25 + y*50
                self.game_pad.blit(src, (ax, ay))
                t = '%d' % i
                r = self.fontObj.render(t, True, (255, 255, 0))
                self.game_pad.blit(r, (ax+5, ay+15))

    def display_point(self):
        mx, my = pygame.mouse.get_pos()
        if mx >= 25 and my >= 25 and mx <= 775 and my <= 775:
            x = np.clip((mx - 25)//50, 0, 14) + 1
            y = np.clip((my - 25)//50, 0, 14) + 1
            if self.env.get_player() == 1:
                color = (0, 0, 0)
            else:
                color = (255, 255, 255)
            pygame.draw.rect(self.game_pad, color, (50*x-10, 50*y-10, 20, 20))

    def display_move(self):
        if len(self.env.get_move_history()) > 0:
            move = self.env.get_move_history()[-1]
            x = move % 15 + 1
            y = move // 15 + 1
            color = (255, 0, 0)
            pygame.draw.rect(self.game_pad, color, (50*x-10, 50*y-10, 20, 20))

    def display_text(self):
        text = [
            'Player %d' % self.env.get_player(),
            'Move %d' % len(self.env.get_move_history()),
            'Winner %d' % self.env.get_winner(),
        ]
        p = [810, 10]
        for t in text:
            r = self.fontObj.render(t, True, (255, 255, 0))
            self.game_pad.blit(r, p)
            p[1] += 40

    def get_move(self):
        mx, my = pygame.mouse.get_pos()
        if mx >= 25 and my >= 25 and mx <= 775 and my <= 775:
            x = np.clip((mx - 25)//50, 0, 14)
            y = np.clip((my - 25)//50, 0, 14)
            action = y*15 + x
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
                if key_button[ord('a')] or key_button[ord('A')]:
                    print(self.env.get_log())
                if key_button[ord('s')] or key_button[ord('S')]:
                    os.makedirs('logs', exist_ok=True)
                    file_name = 'logs/%d.json' % int(time.time())
                    with open(file_name, 'w') as f:
                        json.dump(self.env.get_log(), f, separators=(',', ':'))

        return running

    def __call__(self):
        return self.call()

    def call(self):
        self.display()
        return self.process_event()

    def __del__(self):
        pygame.quit()
