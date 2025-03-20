#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::f32::consts::FRAC_PI_2;
use std::ops::Range;

use bevy::color::palettes::basic::GRAY;
use bevy::core_pipeline::bloom::Bloom;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin};
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::math::{Vec2, Vec3};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, PrimaryWindow};
use itertools::Itertools;
use kdtrees::{Cell3, Octree, OctreeNode};
use rand::Rng;

// Player settings
const SENS: Vec2 = Vec2 { x: 0.003, y: 0.003 };
const PLAYER_MOVE_SPEED: f32 = 50.0;

// Walls and floor position
const SCL: f32 = 100.0;
const X_MIN: f32 = -SCL;
const X_MAX: f32 = SCL;
const Y_MIN: f32 = 0.0;
const Y_MAX: f32 = SCL;
const Z_MIN: f32 = -SCL;
const Z_MAX: f32 = SCL;

// Particle settings
const N_PARTICLES: usize = 10000;
const PARTICLE_SPEED: f32 = 50.0;
const RADIUS: f32 = 0.5;
const DENSITY: f32 = 1.0;

// Octree settings
const TREE_CAPACITY: usize = 10;
const TREE_MAX_DEPTH: Option<usize> = Some(10);
const TREE_BOUNDS: [Range<f32>; 3] = [X_MIN..X_MAX, Y_MIN..Y_MAX, Z_MIN..Z_MAX];

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(ImagePlugin::default_nearest()),
            FpsOverlayPlugin {
                config: FpsOverlayConfig {
                    text_color: Color::srgb(0., 1., 0.),
                    ..default()
                },
            },
        ))
        .insert_resource(OctreeRes(Octree::<f32, Cell3<f32>, Entity>::new(
            TREE_BOUNDS,
            TREE_CAPACITY,
            TREE_MAX_DEPTH,
        )))
        .insert_resource(Flags { draw_octree: true })
        .add_event::<CollisionEvent>()
        .add_systems(Startup, (setup, window_init))
        .add_systems(
            Update,
            (
                apply_velocity,
                find_collisions,
                resolve_collisions.after(find_collisions),
                rotate_player,
                keyboard_input,
                move_player,
                handle_wall_collisions,
                draw_octree.after(find_collisions),
            ),
        )
        .run();
}

#[derive(Component)]
struct Player;

#[derive(Component, Clone)]
struct Velocity(Vec3);

#[derive(Component, Clone)]
struct RigidBody {
    radius: f32,
    mass: f32,
}

#[derive(Bundle)]
struct ParticleBundle {
    transform: Transform,
    velocity: Velocity,
    rigid_body: RigidBody,
    mesh: Mesh3d,
    material: MeshMaterial3d<StandardMaterial>,
}

#[derive(Event, Clone)]
struct CollisionEvent(Entity, Entity);

#[derive(Resource)]
struct OctreeRes(Octree<f32, Cell3<f32>, Entity>);

#[derive(Resource)]
struct Flags {
    draw_octree: bool,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut config_store: ResMut<GizmoConfigStore>,
) {
    let mut rng = rand::rng();

    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.line_width = 0.5;

    // Spawn particles
    let sphere = meshes.add(Sphere::new(RADIUS));
    let particle_material = materials.add(StandardMaterial {
        emissive: LinearRgba::rgb(0., 1., 1.),
        ..default()
    });
    (0..N_PARTICLES).for_each(|_| {
        let x = rng.random_range((0.0 + RADIUS)..(X_MAX - RADIUS));
        let y = rng.random_range((0.0 + RADIUS)..(Y_MAX - RADIUS));
        let z = rng.random_range((0.0 + RADIUS)..(Z_MAX - RADIUS));
        let vx = rng.random_range(0.0..PARTICLE_SPEED);
        let vy = 0.0; //rng.random_range(0.0..PARTICLE_SPEED);
        let vz = 0.0; //rng.random_range(0.0..PARTICLE_SPEED);
        commands.spawn(ParticleBundle {
            transform: Transform::from_xyz(x, y, z),
            velocity: Velocity(Vec3::new(vx, vy, vz)),
            rigid_body: RigidBody {
                radius: RADIUS,
                mass: RADIUS * RADIUS * DENSITY,
            },
            mesh: Mesh3d(sphere.clone()),
            material: MeshMaterial3d(particle_material.clone()),
        });
    });

    // Spawn walls and floor
    let wall_material = materials.add(StandardMaterial {
        base_color: GRAY.into(),
        ..default()
    });
    let floor_material = materials.add(StandardMaterial {
        base_color: GRAY.into(),
        ..default()
    });
    let x_half_width = 0.5 * (X_MAX - X_MIN);
    let y_half_width = 0.5 * (Y_MAX - Y_MIN);
    let z_half_width = 0.5 * (Z_MAX - Z_MIN);
    let x_mid = 0.5 * (X_MAX + X_MIN);
    let y_mid = 0.5 * (Y_MAX + Y_MIN);
    let z_mid = 0.5 * (Z_MAX + Z_MIN);
    let wall_xmin = meshes
        .add(Plane3d::new(Vec3::X, Vec2::new(y_half_width, z_half_width)));
    let wall_xmax = meshes.add(Plane3d::new(
        -Vec3::X,
        Vec2::new(y_half_width, z_half_width),
    ));

    let wall_ymin = meshes
        .add(Plane3d::new(Vec3::Y, Vec2::new(z_half_width, x_half_width)));
    let wall_ymax = meshes.add(Plane3d::new(
        -Vec3::Y,
        Vec2::new(z_half_width, x_half_width),
    ));

    let wall_zmin = meshes
        .add(Plane3d::new(Vec3::Z, Vec2::new(x_half_width, y_half_width)));
    let wall_zmax = meshes.add(Plane3d::new(
        -Vec3::Z,
        Vec2::new(x_half_width, y_half_width),
    ));
    commands.spawn((
        Mesh3d(wall_xmin),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(X_MIN, y_mid, z_mid),
    ));
    commands.spawn((
        Mesh3d(wall_xmax),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(X_MAX, y_mid, z_mid),
    ));
    commands.spawn((
        Mesh3d(wall_ymin),
        MeshMaterial3d(floor_material.clone()),
        Transform::from_xyz(x_mid, Y_MIN, z_mid),
    ));
    commands.spawn((
        Mesh3d(wall_ymax),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(x_mid, Y_MAX, z_mid),
    ));
    commands.spawn((
        Mesh3d(wall_zmin),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(x_mid, y_mid, Z_MIN),
    ));
    commands.spawn((
        Mesh3d(wall_zmax),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(x_mid, y_mid, Z_MAX),
    ));

    // Spawn light
    //commands.spawn((
    //    PointLight {
    //        shadows_enabled: true,
    //        intensity: 10_000_000.,
    //        range: 100.0,
    //        shadow_depth_bias: 0.2,
    //        ..default()
    //    },
    //    Transform::from_xyz(x_mid, Y_MAX - 0.5, z_mid),
    //));

    // Spawn camera
    commands.spawn((
        Player,
        Transform::from_xyz(0.0, 1.0, 0.0),
        Camera3d::default(),
        Camera {
            hdr: true,
            ..default()
        },
        Bloom::NATURAL,
        Tonemapping::TonyMcMapface,
        Projection::from(PerspectiveProjection {
            fov: 70.0_f32.to_radians(),
            ..default()
        }),
    ));
}

fn keyboard_input(
    mut flags: ResMut<Flags>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyB) {
        flags.draw_octree = !flags.draw_octree
    }
}

fn window_init(mut windows: Query<&mut Window, With<PrimaryWindow>>) {
    let Ok(mut window) = windows.get_single_mut() else {
        return;
    };
    window.cursor_options.grab_mode = CursorGrabMode::Confined;
    window.cursor_options.visible = false;
}

fn apply_velocity(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &Velocity)>,
) {
    let dt = time.delta_secs();
    for (mut transform, Velocity(v)) in query.iter_mut() {
        transform.translation += v * dt;
    }
}

fn find_collisions(
    query: Query<(Entity, &RigidBody, &Transform)>,
    mut tree: ResMut<OctreeRes>,
    mut ev_coll: EventWriter<CollisionEvent>,
) {
    tree.0 = Octree::new(TREE_BOUNDS, TREE_CAPACITY, TREE_MAX_DEPTH);
    for (id, RigidBody { radius: r, .. }, transform) in query.iter() {
        let Vec3 { x, y, z } = transform.translation;
        tree.0
            .insert([(x - r, x + r), (y - r, y + r), (z - r, z + r)], id);
    }
    handle_collisions(&tree.0.root, &query, &mut ev_coll);
}

fn handle_collisions(
    node: &OctreeNode<f32, Cell3<f32>, Entity>,
    query: &Query<(Entity, &RigidBody, &Transform)>,
    ev_coll: &mut EventWriter<CollisionEvent>,
) {
    match node {
        OctreeNode::Leaf { ids, .. } => {
            if ids.len() >= 2 {
                for pair in ids.iter().combinations(2) {
                    let id1 = *pair[0];
                    let id2 = *pair[1];
                    let (_, RigidBody { radius: r1, .. }, t1) =
                        query.get(id1).unwrap();
                    let (_, RigidBody { radius: r2, .. }, t2) =
                        query.get(id2).unwrap();
                    if t1.translation.distance_squared(t2.translation)
                        < (r1 + r2) * (r1 + r2)
                    {
                        ev_coll.send(CollisionEvent(id1, id2));
                    }
                }
            }
        }
        OctreeNode::Parent { children, .. } => {
            for child in children.iter() {
                handle_collisions(child, query, ev_coll);
            }
        }
    }
}

fn resolve_collisions(
    mut events: EventReader<CollisionEvent>,
    mut query: Query<(Entity, &RigidBody, &mut Transform, &mut Velocity)>,
) {
    events.read().for_each(|CollisionEvent(id1, id2)| {
        let [(_, rb1, mut t1, mut vel1), (_, rb2, mut t2, mut vel2)] =
            query.many_mut([*id1, *id2]);

        let x1: Vec3 = t1.translation;
        let m1: f32 = rb1.mass;
        let v1: Vec3 = vel1.0;
        let x2: Vec3 = t2.translation;
        let m2: f32 = rb2.mass;
        let v2: Vec3 = vel2.0;

        // Solve for new velocities
        let num: Vec3 = 2.0 * (v1 - v2).dot(x1 - x2) * (x1 - x2);
        let den: f32 = (m1 + m2) * x1.distance_squared(x2);
        let v1new: Vec3 = v1 - m2 * num / den;
        let v2new: Vec3 = v2 + m1 * num / den;
        vel1.0 = v1new;
        vel2.0 = v2new;

        // Handle any overlap
        let r1 = rb1.radius;
        let r2 = rb2.radius;
        let diff = x1 - x2;
        let tangent = diff.normalize();
        let overlap = 0.5 * (r1 + r2 - diff.length());
        t1.translation += overlap * tangent;
        t2.translation -= overlap * tangent;
    });
}

fn rotate_player(
    mouse_motion: Res<AccumulatedMouseMotion>,
    mut player: Query<&mut Transform, With<Player>>,
) {
    let Ok(mut transform) = player.get_single_mut() else {
        return;
    };
    let delta = mouse_motion.delta;
    if delta != Vec2::ZERO {
        let delta_yaw = -delta.x * SENS.x;
        let delta_pitch = -delta.y * SENS.y;

        let (yaw, pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
        let yaw = yaw + delta_yaw;

        const PITCH_LIMIT: f32 = FRAC_PI_2 - 0.01;
        let pitch = (pitch + delta_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);

        transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);
    }
}

fn move_player(
    time: Res<Time>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut player_transform: Single<&mut Transform, With<Player>>,
) {
    let dt = time.delta_secs();
    let forward = player_transform.forward();
    let left = player_transform.left();
    if keyboard_input.pressed(KeyCode::KeyW) {
        player_transform.translation += forward * PLAYER_MOVE_SPEED * dt;
    }
    if keyboard_input.pressed(KeyCode::KeyS) {
        player_transform.translation -= forward * PLAYER_MOVE_SPEED * dt;
    }
    if keyboard_input.pressed(KeyCode::KeyA) {
        player_transform.translation += left * PLAYER_MOVE_SPEED * dt;
    }
    if keyboard_input.pressed(KeyCode::KeyD) {
        player_transform.translation -= left * PLAYER_MOVE_SPEED * dt;
    }
    if keyboard_input.pressed(KeyCode::Space) {
        player_transform.translation += Vec3::Y * PLAYER_MOVE_SPEED * dt;
    }
    if keyboard_input.pressed(KeyCode::ShiftLeft) {
        player_transform.translation -= Vec3::Y * PLAYER_MOVE_SPEED * dt;
    }
}

fn handle_wall_collisions(
    mut particles: Query<(&mut Velocity, &mut Transform, &RigidBody)>,
) {
    particles.par_iter_mut().for_each(
        |(mut velocity, mut transform, RigidBody { radius: r, .. })| {
            let x = transform.translation.x;
            let y = transform.translation.y;
            let z = transform.translation.z;

            if x - r < X_MIN {
                transform.translation.x = X_MIN + r;
                velocity.0.x = -velocity.0.x;
            }
            if x + r > X_MAX {
                transform.translation.x = X_MAX - r;
                velocity.0.x = -velocity.0.x;
            }
            if y - r < Y_MIN {
                transform.translation.y = Y_MIN + r;
                velocity.0.y = -velocity.0.y;
            }
            if y + r > Y_MAX {
                transform.translation.y = Y_MAX - r;
                velocity.0.y = -velocity.0.y;
            }
            if z - r < Z_MIN {
                transform.translation.z = Z_MIN + r;
                velocity.0.z = -velocity.0.z;
            }
            if z + r > Z_MAX {
                transform.translation.z = Z_MAX - r;
                velocity.0.z = -velocity.0.z;
            }
        },
    );
}

fn get_octree_cuboids(
    node: &OctreeNode<f32, Cell3<f32>, Entity>,
) -> Vec<Transform> {
    let mut cuboids = vec![];
    match node {
        OctreeNode::Leaf { ranges, .. } => {
            let [xbounds, ybounds, zbounds] = ranges;
            let xmid = 0.5 * (xbounds.start + xbounds.end);
            let ymid = 0.5 * (ybounds.start + ybounds.end);
            let zmid = 0.5 * (zbounds.start + zbounds.end);
            let xw = xbounds.end - xbounds.start;
            let yw = ybounds.end - ybounds.start;
            let zw = zbounds.end - zbounds.start;
            cuboids.push(
                Transform::from_xyz(xmid, ymid, zmid)
                    .with_scale(Vec3::new(xw, yw, zw)),
            );
        }
        OctreeNode::Parent {
            children, ranges, ..
        } => {
            let [xbounds, ybounds, zbounds] = ranges;
            let xmid = 0.5 * (xbounds.start + xbounds.end);
            let ymid = 0.5 * (ybounds.start + ybounds.end);
            let zmid = 0.5 * (zbounds.start + zbounds.end);
            let xw = xbounds.end - xbounds.start;
            let yw = ybounds.end - ybounds.start;
            let zw = zbounds.end - zbounds.start;
            cuboids.push(
                Transform::from_xyz(xmid, ymid, zmid)
                    .with_scale(Vec3::new(xw, yw, zw)),
            );
            for child in children.iter() {
                cuboids.extend(get_octree_cuboids(child));
            }
        }
    }
    cuboids
}

fn draw_octree(flags: Res<Flags>, mut gizmos: Gizmos, tree: Res<OctreeRes>) {
    if flags.draw_octree {
        let transforms = get_octree_cuboids(&tree.0.root);
        for transform in transforms {
            gizmos.cuboid(transform, Color::srgb(1., 1., 1.));
        }
    }
}
