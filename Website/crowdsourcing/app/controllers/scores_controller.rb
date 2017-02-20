class ScoresController < ApplicationController
  def show_part1
    if session[:img_num].to_i < session[:images].length
      @score = Score.new
      #set image to be shown
      @img = Image.find((session[:images][(session[:img_num].to_i)].to_i))
    else
       redirect_to intro_part2_path
    end 
  end

  def show_part2
      if session[:loc_num].to_i < session[:locs].length
        @score = Score.new
        #set image to be shown
        @loc_id = session[:locs][session[:loc_num].to_i]
        @imgs = Image.where(Image.arel_table[:loc_id]==@loc_id).ids
        @img1 = Image.find(@imgs[0])
        @img2 = Image.find(@imgs[1])
        @img3 = Image.find(@imgs[2])
        @img4 = Image.find(@imgs[3])
      else
         redirect_to end_path
      end
    end

  def create
  #save score value to database (img id, user id, attractiveness, familiarity, uniqueness, friendliness, SAM)
    @score = Score.new(score_params)
    if @score.save
      
      if session[:img_num].to_i < session[:images].length

          redirect_to checkimage_part1_path
      else
          redirect_to end_path
      end 
      
    else
      render '/show_part1'
    end
  end

  def contentcheck_part1
    session[:img_num] = session[:img_num].to_i + 1 #increment image id
    if session[:checkimage] == 1
          redirect_to golden_path
    else
        if session[:img_num].to_i < session[:images].length
          redirect_to show_part1_path
        else
          redirect_to intro_part2_path
        end 
      end 
  end

  def checkimage_part1
      if ((session[:img_num]).to_i % 4) == 3
          session[:checkimage] = 1
      else 
          session[:checkimage] = 0
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
      if ((session[:loc_num]).to_i % 4) == 3
          session[:checkloc] = 1
      else
          session[:checkloc] = 0
      end
      redirect_to contentcheck_part2_path
  end


private

  def score_params
    params.require(:score).permit(:score, :img_id, :user_id, :scale, :recognizability, :lgt)
  end

end
