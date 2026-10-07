// 3D Data Center Scene for HeatWise
const initThreeScene = () => {
    const container = document.getElementById('three-canvas-container');
    if (!container || !window.THREE) return;

    // Avoid duplicate canvases if re-run
    if (container.querySelector('canvas')) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 480;

    // Setup
    const scene = new THREE.Scene();
    
    // Very subtle fog blending with dark/light themes
    const isLightInitial = document.documentElement.classList.contains('light');
    scene.fog = new THREE.FogExp2(isLightInitial ? 0xf1f5f9 : 0x0a0f1d, 0.05);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 4, 12);
    camera.lookAt(0, 0, 0);

    let renderer;
    try {
        renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'high-performance' });
    } catch (e1) {
        try {
            renderer = new THREE.WebGLRenderer({ alpha: true, antialias: false });
        } catch (e2) {
            console.error("WebGL not available in this environment:", e2);
            return;
        }
    }
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(renderer.domElement);

    // Group to hold everything for parallax
    const sceneGroup = new THREE.Group();
    scene.add(sceneGroup);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, isLightInitial ? 0.8 : 0.4);
    sceneGroup.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0x38bdf8, isLightInitial ? 0.8 : 1.0);
    mainLight.position.set(5, 10, 5);
    sceneGroup.add(mainLight);
    
    const fillLight = new THREE.DirectionalLight(0x818cf8, 0.5);
    fillLight.position.set(-5, 5, -5);
    sceneGroup.add(fillLight);

    // Create Racks
    const racks = [];
    const rackGeometry = new THREE.BoxGeometry(1.2, 4, 1.8);
    
    // Status Colors
    const safeColor = new THREE.Color(0x10b981); // Green
    const warnColor = new THREE.Color(0xf59e0b); // Orange
    const critColor = new THREE.Color(0xef4444); // Red
    
    // Create a generic material for the rack chassis
    const chassisMat = new THREE.MeshStandardMaterial({
        color: isLightInitial ? 0x64748b : 0x1e293b,
        roughness: 0.25,
        metalness: 0.75,
    });

    const createRack = (x, z, type) => {
        const rackGroup = new THREE.Group();
        rackGroup.position.set(x, 2, z);
        
        const chassis = new THREE.Mesh(rackGeometry, chassisMat);
        rackGroup.add(chassis);
        
        // Add server slots (emissive strips)
        let emissiveColor = safeColor;
        if (type === 'warn') emissiveColor = warnColor;
        if (type === 'crit') emissiveColor = critColor;
        
        const stripMat = new THREE.MeshStandardMaterial({
            color: emissiveColor,
            emissive: emissiveColor,
            emissiveIntensity: 0.6,
            roughness: 0.4,
        });
        
        // Front face strips
        for (let i = 0; i < 6; i++) {
            const strip = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.1, 0.1), stripMat);
            strip.position.set(0, -1.5 + (i * 0.6), 0.9);
            rackGroup.add(strip);
        }
        
        // Top glow (simulating thermal plume)
        const glowGeo = new THREE.PlaneGeometry(1.5, 2.0);
        const glowMat = new THREE.MeshBasicMaterial({
            color: emissiveColor,
            transparent: true,
            opacity: 0.2,
            blending: THREE.AdditiveBlending,
            side: THREE.DoubleSide
        });
        const glow = new THREE.Mesh(glowGeo, glowMat);
        glow.rotation.x = -Math.PI / 2;
        glow.position.y = 2.01;
        rackGroup.add(glow);

        sceneGroup.add(rackGroup);
        racks.push({ group: rackGroup, type });
        
        return rackGroup;
    };

    // Layout data center racks in 3 staggered rows
    const spacingX = 2.0;
    const spacingZ = 3.0;
    
    // Row 1 (Back)
    createRack(-spacingX * 2, -spacingZ, 'safe');
    createRack(-spacingX, -spacingZ, 'safe');
    createRack(0, -spacingZ, 'crit'); // Hotspot
    createRack(spacingX, -spacingZ, 'warn');
    createRack(spacingX * 2, -spacingZ, 'safe');

    // Row 2 (Middle - Key monitored targets)
    createRack(-spacingX * 2, 0, 'safe');
    const r2 = createRack(-spacingX, 0, 'warn');
    const r3 = createRack(0, 0, 'safe'); 
    createRack(spacingX, 0, 'safe');
    const r5 = createRack(spacingX * 2, 0, 'crit');

    // Row 3 (Front)
    createRack(-spacingX * 1.5, spacingZ, 'safe');
    createRack(0, spacingZ, 'safe');
    createRack(spacingX * 1.5, spacingZ, 'warn');

    // Floor
    const floorGeo = new THREE.PlaneGeometry(32, 32);
    const floorMat = new THREE.MeshStandardMaterial({ 
        color: isLightInitial ? 0xe2e8f0 : 0x0a0f1d, 
        roughness: 0.15, 
        metalness: 0.5 
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0;
    sceneGroup.add(floor);

    // Floor grid (thermal data center coordinate plane)
    const gridHelper = new THREE.GridHelper(32, 32, 0x38bdf8, isLightInitial ? 0x94a3b8 : 0x1e293b);
    gridHelper.position.y = 0.01;
    gridHelper.material.opacity = isLightInitial ? 0.35 : 0.25;
    gridHelper.material.transparent = true;
    sceneGroup.add(gridHelper);

    // Mouse Parallax tracking
    let mouseX = 0;
    let mouseY = 0;
    let targetX = 0;
    let targetY = 0;

    const stepParallax = () => {
        targetX = mouseX * 0.0008;
        targetY = mouseY * 0.0006;
        sceneGroup.rotation.y += 0.05 * (targetX - sceneGroup.rotation.y);
        sceneGroup.rotation.x += 0.05 * (targetY - sceneGroup.rotation.x);
    };

    const onMouseMove = (event) => {
        const halfX = window.innerWidth / 2;
        const halfY = window.innerHeight / 2;
        mouseX = (event.clientX - halfX);
        mouseY = (event.clientY - halfY);
        stepParallax();
        renderer.render(scene, camera);
        updateMarkers();
    };

    document.addEventListener('mousemove', onMouseMove);

    // Project 3D Rack world positions to 2D HTML Overlays
    const pos1 = new THREE.Vector3();
    const pos2 = new THREE.Vector3();
    const pos3 = new THREE.Vector3();

    const updateMarkers = () => {
        const m1 = document.querySelector('.marker-1');
        const m2 = document.querySelector('.marker-2');
        const m3 = document.querySelector('.marker-3');
        
        if (!m1 || !m2 || !m3) return;

        // Obtain world coordinates of the selected racks
        r3.getWorldPosition(pos1);
        pos1.y += 2.3;
        pos1.project(camera);

        r2.getWorldPosition(pos2);
        pos2.y += 2.3;
        pos2.project(camera);

        r5.getWorldPosition(pos3);
        pos3.y += 2.3;
        pos3.project(camera);

        const currentW = container.clientWidth || width;
        const currentH = container.clientHeight || height;

        const toScreen = (pos, el) => {
            const x = (pos.x * 0.5 + 0.5) * currentW;
            const y = (pos.y * -0.5 + 0.5) * currentH;
            
            if (pos.z > 1) {
                el.classList.add('hidden');
            } else {
                el.classList.remove('hidden');
                el.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -100%)`;
            }
        };

        toScreen(pos1, m1);
        toScreen(pos2, m2);
        toScreen(pos3, m3);
    };

    // Animation Loop
    const clock = new THREE.Clock();
    let isRunning = true;

    const animate = () => {
        if (!isRunning) return;
        requestAnimationFrame(animate);

        const t = clock.getElapsedTime();

        // Parallax smooth interpolation
        targetX = mouseX * 0.0008;
        targetY = mouseY * 0.0006;

        sceneGroup.rotation.y += 0.05 * (targetX - sceneGroup.rotation.y);
        sceneGroup.rotation.x += 0.05 * (targetY - sceneGroup.rotation.x);

        // Subtle thermal oscillation
        racks.forEach((rack, i) => {
            rack.group.position.y = 2 + Math.sin(t * 1.8 + i * 0.5) * 0.02;
        });

        renderer.render(scene, camera);
        updateMarkers();
    };

    animate();

    // Resize Handler
    const handleResize = () => {
        if (!container.clientWidth || !container.clientHeight) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Theme Switch Observer
    const applyThemeColors = (isLight) => {
        if (isLight) {
            scene.fog.color.setHex(0xf1f5f9);
            ambientLight.intensity = 0.8;
            mainLight.intensity = 0.8;
            floorMat.color.setHex(0xe2e8f0);
            gridHelper.material.color.setHex(0x94a3b8);
            chassisMat.color.setHex(0x64748b);
        } else {
            scene.fog.color.setHex(0x0a0f1d);
            ambientLight.intensity = 0.4;
            mainLight.intensity = 1.0;
            floorMat.color.setHex(0x0a0f1d);
            gridHelper.material.color.setHex(0x1e293b);
            chassisMat.color.setHex(0x1e293b);
        }
    };

    const themeObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.attributeName === 'class') {
                const isLight = document.documentElement.classList.contains('light');
                applyThemeColors(isLight);
            }
        });
    });
    themeObserver.observe(document.documentElement, { attributes: true });

    // Expose programmatic diagnostics for verification & tests
    window.__HEATWISE_SCENE_STATE__ = {
        scene,
        camera,
        renderer,
        sceneGroup,
        container,
        racks,
        getSceneRotation: () => ({
            x: sceneGroup.rotation.x,
            y: sceneGroup.rotation.y,
            z: sceneGroup.rotation.z
        }),
        getCameraPosition: () => ({
            x: camera.position.x,
            y: camera.position.y,
            z: camera.position.z
        }),
        getMouseParallax: () => ({
            mouseX,
            mouseY,
            targetX,
            targetY,
            rotX: sceneGroup.rotation.x,
            rotY: sceneGroup.rotation.y
        }),
        setManualMouseCoords: (clientX, clientY) => {
            onMouseMove({ clientX, clientY });
        }
    };
};

if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', initThreeScene);
} else {
    initThreeScene();
}
