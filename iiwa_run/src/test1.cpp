#include <ros/ros.h>
#include <geometry_msgs/Pose.h>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>


int main(int argc, char **argv) {
    ros::init(argc, argv, "test1");
    ros::NodeHandle nh;

    // ROS spinning must be running for the MoveGroupInterface to get information
    // about the robot's state. One way to do this is to start an AsyncSpinner
    // beforehand.
    ros::AsyncSpinner spinner(1);
    spinner.start();

    static const std::string PLANNING_GROUP = "All";
    moveit::planning_interface::MoveGroupInterface move_group_interface(PLANNING_GROUP);
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    // RAW Pointer
    const moveit::core::JointModelGroup *joint_model_group =
            move_group_interface.getCurrentState()->getJointModelGroup(PLANNING_GROUP);

    /*ROS_INFO("Available Planning Groups:");
    std::copy(move_group_interface.getJointModelGroupNames().begin(),
              move_group_interface.getJointModelGroupNames().end(),
              std::ostream_iterator<std::string>(std::cout, ", "));*/

    /*const auto joint_names = joint_model_group->getJointModelNames();
    ROS_INFO("Joint Names: ");
    std::copy(joint_names.begin(),
              joint_names.end(),
              std::ostream_iterator<std::string>(std::cout, ", "));*/

    ROS_INFO("Planning in %s.", move_group_interface.getPoseReferenceFrame().c_str());
    move_group_interface.setEndEffector("Needle");
    const auto pose = move_group_interface.getCurrentPose("tool_link");
    ROS_INFO("%s\t%f\t%f\t%f", pose.header.frame_id.c_str(), pose.pose.position.x, pose.pose.position.y,
             pose.pose.position.z);
    geometry_msgs::Pose target_pose1;

    target_pose1.orientation.x = 0.0;
    target_pose1.orientation.y = 0.0;
    target_pose1.orientation.z = 0.0;
    target_pose1.orientation.w = 1;

    target_pose1.position.x = 0.0;
    target_pose1.position.y = 0.2;
    target_pose1.position.z = 1.1;

    /*    target_pose1.orientation.x = -0.686;
    target_pose1.orientation.y = 0.0;
    target_pose1.orientation.z = 0.0;
    target_pose1.orientation.w = -0.727;

    target_pose1.position.x = 0.0;
    target_pose1.position.y = -0.25;
    target_pose1.position.z = 0.3;
     */
    move_group_interface.setPoseTarget(target_pose1);

    // Now, we call the planner to compute the plan and visualize it.
    // Note that we are just planning, not asking move_group_interface
    // to actually move the robot.
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;

    ROS_INFO("Starting planning");
    const bool success = (move_group_interface.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

    ROS_INFO_COND(success, "Plan successful");
    ROS_ERROR_COND(!success, "Plan failed");

    move_group_interface.execute(my_plan);

    ROS_INFO_COND(success, "Plan successful");

    return 0;
}