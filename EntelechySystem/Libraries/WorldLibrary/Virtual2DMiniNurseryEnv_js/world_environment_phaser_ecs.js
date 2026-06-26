import Phaser from 'phaser';

// 定义组件类
class Component {
    constructor() {
        this.entity = null; // 关联的实体
    }
}

// 定义感官组件
class VisionComponent extends Component {
    constructor(range) {
        super();
        this.range = range; // 视野范围
    }
}

class HearingComponent extends Component {
    constructor(sensitivity) {
        super();
        this.sensitivity = sensitivity; // 听觉灵敏度
    }
}

// 定义行为组件
class MovementComponent extends Component {
    constructor(speed) {
        super();
        this.speed = speed; // 移动速度
        this.vx = 0; // 水平速度
        this.vy = 0; // 垂直速度
    }
}

class EmotionComponent extends Component {
    constructor(initialEmotion) {
        super();
        this.emotion = initialEmotion; // 当前表情
    }
}

// 定义实体类
class Entity {
    constructor() {
        this.components = new Map();
    }

    addComponent(component) {
        this.components.set(component.constructor, component);
        component.entity = this;
    }

    getComponent(componentClass) {
        return this.components.get(componentClass);
    }
}

// 定义系统类
class System {
    constructor() {
        this.entities = [];
    }

    addEntity(entity) {
        this.entities.push(entity);
    }

    update(deltaTime) {
        // 子类实现具体逻辑
    }
}

// 定义移动系统
class MovementSystem extends System {
    update(deltaTime) {
        for (const entity of this.entities) {
            const movement = entity.getComponent(MovementComponent);
            if (movement) {
                // 更新实体位置
                movement.entity.x += movement.vx * movement.speed * deltaTime;
                movement.entity.y += movement.vy * movement.speed * deltaTime;

                // 边界检测
                if (movement.entity.x < 0 || movement.entity.x > 800) movement.vx *= -1;
                if (movement.entity.y < 0 || movement.entity.y > 600) movement.vy *= -1;
            }
        }
    }
}

// 初始化游戏
const config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    backgroundColor: '#f0f0f0', // 更改背景颜色为浅灰色
    parent: 'phaser-example',
    scene: {
        preload: preload,
        create: create,
        update: update,
    },
};

const game = new Phaser.Game(config);

let entities = []; // 存储所有实体
let movementSystem;

function preload() {
    // 不加载图片，直接绘制矢量图
}

function create() {
    movementSystem = new MovementSystem();

    // 创建代理人实体
    for (let i = 0; i < 3; i++) {
        const entity = new Entity();

        // 添加移动组件
        const movement = new MovementComponent(1);
        movement.vx = Phaser.Math.FloatBetween(-1, 1);
        movement.vy = Phaser.Math.FloatBetween(-1, 1);
        entity.addComponent(movement);

        // 添加视觉组件
        const vision = new VisionComponent(100);
        entity.addComponent(vision);

        // 添加情感组件
        const emotion = new EmotionComponent('neutral');
        entity.addComponent(emotion);

        // 创建图形对象并绑定到实体
        const graphics = this.add.graphics();
        graphics.fillStyle(0x0000ff, 1); // 蓝色
        graphics.fillCircle(0, 0, 10); // 半径为10的圆
        entity.graphics = graphics;

        // 随机初始化位置
        entity.x = Phaser.Math.Between(50, 750);
        entity.y = Phaser.Math.Between(50, 550);
        entity.graphics.setPosition(entity.x, entity.y);

        entities.push(entity);
        movementSystem.addEntity(entity);
    }
}

function update(time, delta) {
    const deltaTime = delta / 1000;
    movementSystem.update(deltaTime);

    // 更新实体的图形位置
    for (const entity of entities) {
        const movement = entity.getComponent(MovementComponent);
        if (movement) {
            entity.x += movement.vx;
            entity.y += movement.vy;

            // 确保实体不会移出屏幕边界
            entity.x = Phaser.Math.Clamp(entity.x, 0, 800);
            entity.y = Phaser.Math.Clamp(entity.y, 0, 600);

            entity.graphics.setPosition(entity.x, entity.y);
        }
    }
}