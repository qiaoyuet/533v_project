import os


def log_values(cost, dists, penalties, lam, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts, result_path):
    avg_cost = cost.mean().item()
    avg_dist = dists.mean().item()
    avg_pen = penalties.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}, avg_dist: {}, avg_pen: {}, lambda: {}'.format(
        epoch, batch_id, avg_cost, avg_dist, avg_pen, lam))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    log_items = [
        epoch, batch_id, avg_cost, avg_dist, avg_pen, lam
    ]
    log_str = '{:.8f}_' * (len(log_items) - 1) + '{:.8f}'
    formatted_log = log_str.format(*log_items)

    res_file = os.path.join(result_path, 'res_train.txt')
    with open(res_file, 'a+') as f:
        f.write(formatted_log)
        f.write('\n')

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)
