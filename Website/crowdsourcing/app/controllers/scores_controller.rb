class ScoresController < ApplicationController
  def show_part1
    @title = "Task - Part 1"
    if session[:checkimage]==1
        @score = Score.new

        #get a golden image
        @golden_question = GoldenQuestion.find(session[:goldens][(session[:golden_num].to_i)])
        @imgid = @golden_question.img_id
        @options = @golden_question.options
        @img = Image.find(@imgid)
        session[:golden_num] = session[:golden_num].to_i + 1
        if(session[:golden_num] > session[:goldens].length-1)
            session[:golden_num] = 0
        end
        @log_code = "1|"+@golden_question.img_id.to_s+"|"+session[:userid].to_s;
    elsif session[:img_num].to_i < session[:images].length
      @score = Score.new
      #set image to be shown
      @img = Image.find((session[:images][(session[:img_num].to_i)].to_i))
      @imgid = (session[:images][(session[:img_num].to_i)].to_i);
      @log_code = "1|"+@imgid.to_s+"|"+session[:userid].to_s;
    else
       redirect_to intro_part2_path
    end 
  end

  def show_part2
      @title = "Task - Part 2"
      if session[:loc_num].to_i < session[:locs].length
        @score = Score.new
        #set image to be shown
        @loc_id = session[:locs][(session[:loc_num].to_i)].to_i
        @imgs = Image.where(loc_id: @loc_id).ids
        @img1 = Image.find(@imgs[0])
        @img2 = Image.find(@imgs[1])
        @img3 = Image.find(@imgs[2])
        @img4 = Image.find(@imgs[3])
        @log_code = "2|"+@loc_id.to_s+"|"+session[:userid].to_s;
      else
         redirect_to end_path
      end
    end

  def create
  #save score value to database
    if session[:checkimage]==1
      #save answer of golden question

    end

    @score = Score.new(score_params)
    @score.end_time = Time.now.strftime("%H:%M:%S %z")
    if @score.save
        if session[:part] == 1
          if session[:img_num].to_i < session[:images].length
              redirect_to checkimage_part1_path
          else
              redirect_to intro_part2_path
          end
        else
          if session[:loc_num].to_i < session[:locs].length
              session[:loc_num] = session[:loc_num].to_i + 1 #increment loc id
              redirect_to show_part2_path
          else
              redirect_to end_path
          end
        end
    else
        if session[:part] == 1
            render '/show_part1'
        else
            render '/show_part2'
        end
    end
  end

  def contentcheck_part1
    session[:img_num] = session[:img_num].to_i + 1 #increment image id
    if session[:checkimage] == 1
          redirect_to show_part1_path
    else
        if session[:img_num].to_i < session[:images].length
          redirect_to show_part1_path
        else
          redirect_to intro_part2_path
        end 
      end 
  end

  def checkimage_part1
      if session[:checkimage] == 1
         session[:img_num] = session[:img_num].to_i - 1 #backtrack
         session[:checkimage] = 0
      else
          if ((session[:img_num]).to_i == 10) || ((session[:img_num]).to_i == 20) || ((session[:img_num]).to_i == 40) || ((session[:img_num]).to_i == 60) || ((session[:img_num]).to_i == 70)
               session[:checkimage] = 1
          else
               session[:checkimage] = 0
          end
      end
      redirect_to contentcheck_part1_path
  end

  def contentcheck_part2
    session[:loc_num] = session[:loc_num].to_i + 1 #increment image id
    if session[:checkloc] == 1
          redirect_to golden_path
    else
        if session[:loc_num].to_i < session[:locs].length
          redirect_to show_part2_path
        else
          redirect_to end_path
        end
      end
  end

  def checkimage_part2
      if ((session[:loc_num]).to_i == 10) || ((session[:loc_num]).to_i == 20) || ((session[:loc_num]).to_i == 40) || ((session[:loc_num]).to_i == 50) || ((session[:loc_num]).to_i == 70)
          session[:checkloc] = 1
      else
          session[:checkloc] = 0 
      end
      redirect_to contentcheck_part2_path
  end


private

  def score_params
    params.require(:score).permit(:user_id, :part, :loc_id, :img_id, :attractiveness, :familiarity, :uniqueness, :friendliness, :pleasure, :arousal, :dominance, :golden_answer, :start_time)
  end

  def golden_params
    params.require()
  end

end
