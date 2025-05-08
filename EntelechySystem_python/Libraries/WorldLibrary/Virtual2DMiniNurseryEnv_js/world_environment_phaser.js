import Phaser from 'phaser';

// 定义全局变量
let agents = [];
let landmarks = [];
let sprites = [];

// 定义 Phaser 配置
const config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    backgroundColor: '#ffffff',
    parent: 'phaser-example',
    scene: {
        preload: preload,
        create: create,
        update: update,
    },
};

// 创建 Phaser 游戏实例
const game = new Phaser.Game(config);

function preload() {
    // 不加载图片，直接绘制矢量图
}

function create() {
    // 创建代理人
    for (let i = 0; i < 3; i++) {
        const agent = {
            x: Phaser.Math.Between(100, 700),
            y: Phaser.Math.Between(100, 700),
            vx: Phaser.Math.FloatBetween(-1, 1),
            vy: Phaser.Math.FloatBetween(-1, 1),
        };
        agents.push(agent);

        // 绘制圆形作为代理人
        const circle = this.add.circle(agent.x, agent.y, 20, 0x3498db); // 半径20，颜色蓝色
        sprites.push(circle);
    }

    // 创建地标
    for (let i = 0; i < 3; i++) {
        const landmark = {
            x: Phaser.Math.Between(100, 700),
            y: Phaser.Math.Between(100, 700),
        };
        landmarks.push(landmark);

        // 绘制圆形作为地标
        const circle = this.add.circle(landmark.x, landmark.y, 15, 0xe74c3c); // 半径15，颜色红色
        sprites.push(circle);
    }
}

function update() {
    // 更新代理人位置
    for (let i = 0; i < agents.length; i++) {
        const agent = agents[i];
        agent.x += agent.vx;
        agent.y += agent.vy;

        // 边界检测
        if (agent.x < 0 || agent.x > 800) agent.vx *= -1;
        if (agent.y < 0 || agent.y > 600) agent.vy *= -1;

        // 更新对应的图形位置
        sprites[i].x = agent.x;
        sprites[i].y = agent.y;
    }
}