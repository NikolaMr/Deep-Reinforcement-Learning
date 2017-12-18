import DDQN_shell

game=DDQN_shell.Wrapped_Game(DDQN_shell.env)

model = DDQN_shell.build_model()
model.load_weights('./backup_pong_lr-4/' + '150000_iters_pdv4_ddqn_lr-4_tmr_100_after_500000.h5')

DDQN_shell.train_model(model, game)

