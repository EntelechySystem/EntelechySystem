import Phaser from 'phaser';

// 初始化游戏
const config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    backgroundColor: '#f0f0f0', // 浅灰色背景
    parent: 'phaser-example',
    scene: {
        preload: preload,
        create: create,
        update: update,
    },
};

const game = new Phaser.Game(config);

let agents = []; // 存储所有代理人

function preload() {
    // 不加载图片，直接绘制矢量图
}

function create() {
    // 创建代理人
    for (let i = 0; i < 3; i++) {
        // 创建图形对象
        const graphics = this.add.graphics();
        graphics.fillStyle(0x0000ff, 1); // 蓝色
        graphics.fillCircle(0, 0, 10); // 半径为10的圆

        // 随机初始化位置
        const x = Phaser.Math.Between(50, 750);
        const y = Phaser.Math.Between(50, 550);
        graphics.setPosition(x, y);

        // 随机初始化速度
        const vx = Phaser.Math.FloatBetween(-100, 100);
        const vy = Phaser.Math.FloatBetween(-100, 100);

        // 存储代理人信息
        agents.push({ graphics, x, y, vx, vy });
    }
}

function update(time, delta) {
    const deltaTime = delta / 1000; // 将delta转换为秒

    for (const agent of agents) {
        // 更新位置
        agent.x += agent.vx * deltaTime;
        agent.y += agent.vy * deltaTime;

        // 边界检测并反弹
        if (agent.x < 0 || agent.x > 800) {
            agent.vx *= -1;
            agent.x = Phaser.Math.Clamp(agent.x, 0, 800);
        }
        if (agent.y < 0 || agent.y > 600) {
            agent.vy *= -1;
            agent.y = Phaser.Math.Clamp(agent.y, 0, 600);
        }

        // 更新图形位置
        agent.graphics.setPosition(agent.x, agent.y);
    }
}