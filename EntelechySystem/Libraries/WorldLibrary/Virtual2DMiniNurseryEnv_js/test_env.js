import Phaser from 'phaser';
import MainScene from './world_environment.js';

// 配置 Phaser 游戏
const config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    backgroundColor: '#ffffff',
    parent: 'phaser-example',
    scene: [MainScene],
};

// 启动游戏
const game = new Phaser.Game(config);